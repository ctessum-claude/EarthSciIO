//! A known process environment for the duration of a test.
//!
//! Shared rather than per-module because the process environment is one object
//! and `cargo test` runs modules' tests on the same threads: a lock private to
//! one module cannot stop another module's test from setting `AWS_REGION`
//! underneath it. Every test in this crate that reads a variable through
//! [`store_options_from_env`](crate::store_options_from_env),
//! [`resolve_region`](crate::transport::resolve_region) or
//! [`BUCKET_OPTIONS_ENV`](crate::BUCKET_OPTIONS_ENV) takes this guard.

/// Variable prefixes an [`EnvScope`] clears and restores: everything this crate
/// harvests, plus its own `EARTHSCI_S3_*` statements.
const SCOPED_PREFIXES: [&str; 4] = ["AWS_", "GOOGLE_", "AZURE_", "EARTHSCI_S3_"];

/// Serializes the tests that mutate the process environment, so one test's
/// `AWS_*` variable cannot appear in another test's harvest.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A known environment for the duration of a scope, restored on the way out
/// whether the test passes or panics. Holds [`ENV_LOCK`] while it lives.
///
/// It **removes every scoped variable first**, because these tests are about
/// what the ambient environment does to a read: one left over from the
/// developer's shell (`AWS_PROFILE`, a real `AWS_ACCESS_KEY_ID`) would otherwise
/// decide the outcome, which is the very failure mode under test.
pub(crate) struct EnvScope {
    _guard: std::sync::MutexGuard<'static, ()>,
    restore: Vec<(String, String)>,
    clear: Vec<String>,
}

impl EnvScope {
    pub(crate) fn new(vars: &[(&str, &str)]) -> Self {
        let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let restore: Vec<(String, String)> = std::env::vars()
            .filter(|(k, _)| SCOPED_PREFIXES.iter().any(|p| k.starts_with(p)))
            .collect();
        for (k, _) in &restore {
            std::env::remove_var(k);
        }
        for (k, v) in vars {
            std::env::set_var(k, v);
        }
        Self {
            _guard: guard,
            restore,
            clear: vars.iter().map(|(k, _)| (*k).to_string()).collect(),
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for k in &self.clear {
            std::env::remove_var(k);
        }
        for (k, v) in &self.restore {
            std::env::set_var(k, v);
        }
    }
}
