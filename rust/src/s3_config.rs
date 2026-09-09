//! What a deployment says about the S3 buckets it reads — **per bucket**.
//!
//! Everything this crate knew about S3 configuration used to be one answer per
//! process: one `AWS_*` harvest
//! ([`store_options_from_env`](crate::store_options_from_env)), one region
//! ([`resolve_region`](crate::transport::resolve_region)), one signed-bucket
//! list ([`SIGNED_BUCKETS_ENV`](crate::transport::SIGNED_BUCKETS_ENV)). That is
//! enough for as long as every `s3://` a run touches wants the same answer, and
//! it stops being enough the moment two of them do not — a public store in one
//! region beside a **requester-pays** store in another, or somebody else's
//! bucket beside our own private one.
//!
//! [`BUCKET_OPTIONS_ENV`] is the general form of the statement the signed-bucket
//! list already makes: *name the bucket, state the intent*. It carries
//! `object_store`'s own option keys, so it needs no vocabulary of its own and
//! nothing here has to know what any particular key means:
//!
//! ```text
//! EARTHSCI_S3_BUCKET_OPTIONS={
//!   "inmap-model": {
//!     "aws_skip_signature": "false",
//!     "aws_request_payer":  "true",
//!     "aws_region":         "us-east-2"
//!   }
//! }
//! ```
//!
//! That example is the case it was written for. A requester-pays bucket refuses
//! anonymous requests outright, so reading one needs SigV4 **and**
//! `x-amz-request-payer: requester` on every request — and it needs them for
//! *that* bucket without turning them on for every other `s3://` the same
//! process touches, because the process is also writing its own output to a
//! bucket it authenticates to differently.
//!
//! ## Precedence, and why it sits where it does
//!
//! A per-bucket statement beats the **environment** and loses to the **caller**:
//!
//! ```text
//! store_options_from_env()  <  EARTHSCI_S3_BUCKET_OPTIONS  <  the caller's own options
//! ```
//!
//! The environment is ambient — a platform-injected role, a developer's shell —
//! and naming a bucket is deliberate, so the named thing wins. A caller passing
//! [`DataSource::store_options`](crate::DataSource::store_options) is more
//! deliberate still, and is describing one loader rather than one deployment.
//!
//! ## It cannot make a public read start signing by accident
//!
//! The property [`transport::s3`](crate::transport) guards most carefully is
//! that ambient credentials must never turn an anonymous read of a public
//! bucket into a signed one — that failure mode has cost this codebase two
//! rounds of debugging, and it presents as a 403 on a bucket that is readable by
//! anyone. Nothing here weakens it. A bucket has to be **named**, there is no
//! wildcard, and a bucket named only to pin its region reads exactly as
//! anonymously as it did before: see [`stated_signing`], which reports what the
//! options say and `None` when they say nothing.

use std::collections::BTreeMap;

/// Per-bucket `object_store` options, as a JSON object of
/// `{"<bucket>": {"<option key>": "<value>"}}`. See the module note.
///
/// Bucket names only — an entry is matched against the `s3://<bucket>/…` host,
/// so it cannot accidentally describe some other origin — and no wildcard, for
/// the same reason [`SIGNED_BUCKETS_ENV`](crate::transport::SIGNED_BUCKETS_ENV)
/// has none.
pub const BUCKET_OPTIONS_ENV: &str = "EARTHSCI_S3_BUCKET_OPTIONS";

/// Option keys that are a *statement about signing* rather than a credential.
/// Honoured verbatim wherever they come from, environment included: no platform
/// injects `AWS_SKIP_SIGNATURE`, so it can only have been written by whoever
/// deployed this process. `=false` is how a caller asks for signed access.
pub const S3_SIGNING_KEYS: [&str; 2] = ["aws_skip_signature", "skip_signature"];

/// Option keys that mean "there is a way to authenticate to S3" — key material,
/// the ambient-role pointers a container platform sets, and the assume-role
/// inputs. Every spelling `object_store`'s own `AmazonS3ConfigKey::from_str`
/// accepts, because it takes a prefixed *and* an unprefixed form of nearly all
/// of them and a list naming only one of a pair is a list with a hole in it: the
/// unnamed spelling configures a credential that this module then fails to
/// notice, so a read the caller asked to sign goes out anonymous.
///
/// Presence is not by itself consent: see
/// [`read_store_options`](crate::read_store_options) for who has to have said it
/// before signing stays on.
pub const S3_CREDENTIAL_KEYS: [&str; 16] = [
    "aws_access_key_id",
    "access_key_id",
    "aws_secret_access_key",
    "secret_access_key",
    "aws_session_token",
    "session_token",
    "aws_token",
    "token",
    "aws_container_credentials_relative_uri",
    "container_credentials_relative_uri",
    "aws_container_credentials_full_uri",
    "container_credentials_full_uri",
    "aws_web_identity_token_file",
    "web_identity_token_file",
    "aws_role_arn",
    "role_arn",
];

/// Static key material, the half of [`S3_CREDENTIAL_KEYS`] an operator types out
/// rather than a platform injecting it.
pub const S3_STATIC_KEY_KEYS: [&str; 2] = ["aws_access_key_id", "access_key_id"];

/// Endpoint-override keys, in every spelling `object_store` parses. An endpoint
/// is never injected by a platform: it is always somebody pointing `s3://` at a
/// specific S3-compatible deployment.
pub const S3_ENDPOINT_KEYS: [&str; 5] = [
    "endpoint",
    "endpoint_url",
    "aws_endpoint",
    "aws_endpoint_url",
    "aws_endpoint_url_s3",
];

/// Region keys, both spellings `object_store` accepts.
pub const S3_REGION_KEYS: [&str; 2] = ["aws_region", "region"];

/// Does `options` carry any of `keys`?
#[must_use]
pub fn has_any(options: &[(String, String)], keys: &[&str]) -> bool {
    options.iter().any(|(key, _)| keys.contains(&key.as_str()))
}

/// The value `options` gives `key`, if any.
fn value_of<'a>(options: &'a [(String, String)], keys: &[&str]) -> Option<&'a str> {
    options
        .iter()
        .find(|(k, _)| keys.contains(&k.as_str()))
        .map(|(_, v)| v.as_str())
}

/// The bucket an `s3://` / `s3a://` URL addresses, lowercased.
///
/// `None` for every other scheme, so a caller can ask this of any URL and get
/// "no per-bucket statement applies" rather than having to test the scheme
/// first. Bucket names are already lowercase, so folding case cannot merge two
/// distinct buckets; it only stops a mis-cased entry from silently matching
/// nothing, which would present as an unexplained 403.
#[must_use]
pub fn bucket_of_url(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("s3://")
        .or_else(|| url.strip_prefix("s3a://"))?;
    let bucket = rest.split('/').next().unwrap_or("");
    (!bucket.is_empty()).then(|| bucket.to_ascii_lowercase())
}

/// Parse a [`BUCKET_OPTIONS_ENV`] spec.
///
/// Blank is treated as unset — an empty string is how a secrets manager spells
/// "not configured", and reading it as a value would be a map with one nameless
/// bucket in it.
///
/// Values may be given as JSON strings, booleans or numbers, because the spec is
/// usually rendered by a tool (Terraform's `jsonencode` of a typed map) rather
/// than typed by hand, and `true` is what such a tool emits for a flag. They are
/// handed to `object_store` as strings either way. Anything else — a nested
/// object, an array, `null` — is an error rather than a skipped key, so a
/// misspelt shape fails loudly instead of reading as "no options for this
/// bucket" and going out anonymous.
///
/// # Errors
///
/// The spec is not a JSON object, a bucket's value is not a JSON object, or an
/// option value is not a string, boolean or number.
pub fn parse_bucket_options(spec: &str) -> Result<BTreeMap<String, Vec<(String, String)>>, String> {
    if spec.trim().is_empty() {
        return Ok(BTreeMap::new());
    }
    let root: serde_json::Value =
        serde_json::from_str(spec).map_err(|e| format!("not valid JSON: {e}"))?;
    let buckets = root
        .as_object()
        .ok_or_else(|| "expected a JSON object of {\"bucket\": {options}}".to_string())?;

    let mut out = BTreeMap::new();
    for (bucket, options) in buckets {
        let bucket = bucket.trim().to_ascii_lowercase();
        if bucket.is_empty() {
            return Err("a bucket name is empty".to_string());
        }
        let options = options
            .as_object()
            .ok_or_else(|| format!("'{bucket}' does not map to a JSON object of options"))?;
        let mut pairs = Vec::with_capacity(options.len());
        for (key, value) in options {
            let value = match value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Number(n) => n.to_string(),
                other => {
                    return Err(format!(
                        "'{bucket}'.'{key}' is {}, and an option value must be a string, boolean or number",
                        kind_of(other)
                    ))
                }
            };
            pairs.push((key.trim().to_ascii_lowercase(), value));
        }
        out.insert(bucket, pairs);
    }
    Ok(out)
}

/// A JSON value's kind, for an error message.
fn kind_of(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// The whole per-bucket map from the environment, empty when unset or blank.
///
/// # Errors
///
/// [`BUCKET_OPTIONS_ENV`] is set to something [`parse_bucket_options`] refuses.
/// Callers that can fail should, because a deployment that mis-renders this
/// variable has no other way to find out: the alternative — treating a
/// malformed spec as no spec — is a read that quietly goes out anonymous and
/// only fails much later, against whichever bucket needed the statement.
pub fn bucket_options_from_env() -> Result<BTreeMap<String, Vec<(String, String)>>, String> {
    match std::env::var(BUCKET_OPTIONS_ENV) {
        Ok(spec) => parse_bucket_options(&spec),
        Err(_) => Ok(BTreeMap::new()),
    }
}

/// The options [`BUCKET_OPTIONS_ENV`] states for `bucket`, empty when it names
/// no such bucket — and empty, rather than an error, when the spec is malformed.
///
/// The infallible spelling, for the option-resolution path, which returns a
/// `Vec` and has nowhere to put a failure. The transports call
/// [`bucket_options_from_env`] instead and refuse the fetch, so a malformed spec
/// is reported at the first read rather than swallowed here.
#[must_use]
pub fn options_for_bucket(bucket: &str) -> Vec<(String, String)> {
    bucket_options_from_env()
        .unwrap_or_default()
        .remove(&bucket.to_ascii_lowercase())
        .unwrap_or_default()
}

/// The options [`BUCKET_OPTIONS_ENV`] states for the bucket `url` addresses.
/// Empty for a non-`s3://` URL, so this is safe to ask of anything.
#[must_use]
pub fn options_for_url(url: &str) -> Vec<(String, String)> {
    bucket_of_url(url)
        .map(|b| options_for_bucket(&b))
        .unwrap_or_default()
}

/// What `options` say about signing: `Some(true)` sign, `Some(false)` read
/// anonymously, `None` they do not say.
///
/// An explicit `aws_skip_signature` is the answer whenever it is present, in
/// either polarity. Failing that, a credential means signing — that is what a
/// credential is *for* in a set of options somebody wrote deliberately. Note the
/// asymmetry with the environment, which is deliberate: an ambient credential
/// proves only that the platform injected a role for something else, whereas one
/// written beside a bucket's name is about that bucket.
#[must_use]
pub fn stated_signing(options: &[(String, String)]) -> Option<bool> {
    if let Some(value) = value_of(options, &S3_SIGNING_KEYS) {
        // `object_store` parses these as booleans and refuses anything else; be
        // liberal here so a `"1"` or a `"TRUE"` is not read as its opposite.
        return Some(!matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "true" | "1" | "yes" | "on"
        ));
    }
    has_any(options, &S3_CREDENTIAL_KEYS).then_some(true)
}

/// The region `options` state, if they state one.
#[must_use]
pub fn stated_region(options: &[(String, String)]) -> Option<&str> {
    value_of(options, &S3_REGION_KEYS).filter(|r| !r.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(map: &BTreeMap<String, Vec<(String, String)>>, bucket: &str) -> Vec<(String, String)> {
        map.get(bucket).cloned().unwrap_or_default()
    }

    #[test]
    fn an_unset_or_blank_spec_is_no_buckets() {
        assert!(parse_bucket_options("").unwrap().is_empty());
        assert!(parse_bucket_options("   ").unwrap().is_empty());
        assert!(parse_bucket_options("{}").unwrap().is_empty());
    }

    #[test]
    fn a_bucket_carries_its_own_options() {
        let map = parse_bucket_options(
            r#"{"inmap-model":{"aws_request_payer":"true","aws_region":"us-east-2"}}"#,
        )
        .unwrap();
        let opts = pairs(&map, "inmap-model");
        assert_eq!(value_of(&opts, &["aws_request_payer"]), Some("true"));
        assert_eq!(stated_region(&opts), Some("us-east-2"));
    }

    #[test]
    fn a_tool_rendered_boolean_is_a_value_and_not_a_refusal() {
        // Terraform's `jsonencode` of a typed map emits `true`, not `"true"`.
        let map = parse_bucket_options(r#"{"b":{"aws_request_payer":true,"n":3}}"#).unwrap();
        let opts = pairs(&map, "b");
        assert_eq!(value_of(&opts, &["aws_request_payer"]), Some("true"));
        assert_eq!(value_of(&opts, &["n"]), Some("3"));
    }

    #[test]
    fn bucket_names_and_option_keys_are_folded_to_lowercase() {
        let map = parse_bucket_options(r#"{"InMap-Model":{"AWS_Region":"us-east-2"}}"#).unwrap();
        let opts = pairs(&map, "inmap-model");
        assert_eq!(stated_region(&opts), Some("us-east-2"));
    }

    #[test]
    fn a_malformed_spec_is_an_error_rather_than_an_empty_map() {
        // Each of these would otherwise read as "no options for this bucket",
        // which is a read that silently goes out anonymous.
        assert!(parse_bucket_options("not json").is_err());
        assert!(parse_bucket_options(r#"["inmap-model"]"#).is_err());
        assert!(parse_bucket_options(r#"{"inmap-model":"us-east-2"}"#).is_err());
        assert!(parse_bucket_options(r#"{"inmap-model":{"k":null}}"#).is_err());
        assert!(parse_bucket_options(r#"{"inmap-model":{"k":["a"]}}"#).is_err());
        assert!(parse_bucket_options(r#"{"":{"k":"v"}}"#).is_err());
    }

    #[test]
    fn the_bucket_of_a_url_is_its_host_and_nothing_else() {
        assert_eq!(
            bucket_of_url("s3://inmap-model/a/b"),
            Some("inmap-model".into())
        );
        assert_eq!(
            bucket_of_url("s3a://Inmap-Model/a"),
            Some("inmap-model".into())
        );
        // No key is still a bucket: `s3://bucket` addresses the bucket itself.
        assert_eq!(
            bucket_of_url("s3://inmap-model"),
            Some("inmap-model".into())
        );
        assert_eq!(bucket_of_url("https://example.com/a"), None);
        assert_eq!(bucket_of_url("s3:///a"), None);
    }

    #[test]
    fn signing_is_stated_by_a_polarity_or_by_a_credential() {
        let sign = [("aws_skip_signature".to_string(), "false".to_string())];
        let anon = [("aws_skip_signature".to_string(), "true".to_string())];
        let creds = [("aws_access_key_id".to_string(), "AKIA…".to_string())];
        let region_only = [("aws_region".to_string(), "eu-west-1".to_string())];
        assert_eq!(stated_signing(&sign), Some(true));
        assert_eq!(stated_signing(&anon), Some(false));
        assert_eq!(stated_signing(&creds), Some(true));
        // The property that matters most: naming a bucket to pin its region
        // says NOTHING about signing, so a public read stays public.
        assert_eq!(stated_signing(&region_only), None);
        assert_eq!(stated_signing(&[]), None);
    }

    #[test]
    fn a_truthy_skip_signature_is_not_read_as_its_opposite() {
        for spelling in ["true", "TRUE", " True ", "1", "yes", "on"] {
            let opts = [("aws_skip_signature".to_string(), spelling.to_string())];
            assert_eq!(stated_signing(&opts), Some(false), "{spelling}");
        }
    }
}
