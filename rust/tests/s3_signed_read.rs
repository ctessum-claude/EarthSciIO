//! Reading a **private** S3 object through the cache, signed.
//!
//! The gap this closes, stated as the failure it produces: every whole-file
//! format — `shapefile`, `ff10`, `netcdf`, `geotiff` — reads through
//! `transport::s3`, which is an anonymous `s3://` → regional-HTTPS rewriter. A
//! `.zip` in a private bucket is therefore unreadable no matter what the
//! reader's IAM role grants, and it fails as `HTTP 403` partway into a run.
//! `DataSource::store_access` / `store_options`, which is where a loader would
//! ask for a signed read, are store-backed (Zarr) only and are documented as
//! "ignored by whole-file readers".
//!
//! Two tiers of evidence, and the first is why the second may be skipped.
//!
//! [`nothing_signs_unless_a_bucket_is_named`] runs everywhere and offline. It is
//! the regression that matters most: the default must stay anonymous, because a
//! transport that started signing on ambient credentials would sign a read of a
//! PUBLIC bucket and turn it into a 403 — the exact bug EarthSciIO's U6 note
//! records fixing on the `object_store` path.
//!
//! [`a_named_private_bucket_reads_signed`] is the real thing: a live GET of a
//! private object, anonymous first (expecting the 403 that motivates all of
//! this) and then signed. It needs AWS credentials and the network, so it is
//! opt-in through `EARTHSCI_S3_SIGNED_TEST`, whose value is the `s3://` URL of a
//! private object the caller can read.

use std::path::PathBuf;

use earthsciio::transport::{S3Transport, Transport};
use earthsciio::transport::Conditional;

/// Serializes the tests in this file.
///
/// `S3Transport::new()` reads the process environment, and one test here *sets*
/// `EARTHSCI_S3_BUCKET_OPTIONS` — so without this, that test decides whether the
/// tests asserting "nothing is signed by default" see a named bucket. Every test
/// takes it, including the offline ones, because which of them run together is
/// not something a test may assume.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take [`ENV_LOCK`], surviving a previous test's panic.
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A staging path on a throwaway directory, as the cache would hand a transport.
fn staging() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let dest = dir.path().join("staged.part");
    (dir, dest)
}

/// **The default is anonymous, and nothing infers otherwise.**
///
/// Asserted against a bucket that is genuinely public (`inmap-model`, the InMAP
/// ISRM store) so the claim is about the transport's decision and not about
/// whether some credential happens to exist: whatever this machine's
/// environment is carrying, an unnamed bucket must not be signed.
#[test]
fn nothing_signs_unless_a_bucket_is_named() {
    let _lock = env_lock();
    let t = S3Transport::new();
    assert!(
        !t.signs("inmap-model"),
        "a public bucket must never sign — that is a 403 dressed as a config change"
    );

    let named = S3Transport::new().signing_buckets(["some-private-bucket"]);
    assert!(named.signs("some-private-bucket"));
    assert!(
        !named.signs("inmap-model"),
        "naming one bucket must not widen to any other"
    );
}

/// The public read still works, unchanged, through the anonymous path.
///
/// Kept beside the signing test rather than in the zarr suite because the thing
/// at risk is this transport's *routing*: a bug in `signs()` shows up here as a
/// public object that suddenly needs credentials.
#[test]
fn a_public_bucket_still_reads_anonymously() {
    let _lock = env_lock();
    if std::env::var("EARTHSCI_S3_ONLINE_TEST").is_err() {
        eprintln!("set EARTHSCI_S3_ONLINE_TEST=1 to run the live public read");
        return;
    }
    let (_dir, dest) = staging();
    let t = S3Transport::new();
    let out = t
        .fetch(
            "s3://inmap-model/isrm_v1.2.2.zarr/.zmetadata",
            &dest,
            &Conditional::default(),
            None,
        )
        .expect("the public ISRM store reads with no credentials at all");
    assert!(out.bytes_written > 0);
}

/// **A named private bucket reads, and the same object refuses anonymously.**
///
/// Both halves matter. The signed read alone would pass just as well against a
/// public object, proving nothing; the anonymous 403 beside it is what shows the
/// object really is private and that signing is what changed the answer.
#[test]
fn a_named_private_bucket_reads_signed() {
    let _lock = env_lock();
    let Ok(url) = std::env::var("EARTHSCI_S3_SIGNED_TEST") else {
        eprintln!(
            "set EARTHSCI_S3_SIGNED_TEST=s3://<bucket>/<key> (a PRIVATE object you \
             can read) plus AWS credentials to run this"
        );
        return;
    };
    let bucket = url
        .strip_prefix("s3://")
        .and_then(|r| r.split('/').next())
        .expect("EARTHSCI_S3_SIGNED_TEST must be an s3:// URL")
        .to_string();

    // Anonymous first: this is the failure the feature exists to fix, and
    // asserting it here keeps the test honest if the object is ever made public.
    let (_d1, dest1) = staging();
    let anonymous = S3Transport::new().signing_buckets(Vec::<String>::new());
    let refused = anonymous.fetch(&url, &dest1, &Conditional::default(), None);
    assert!(
        refused.is_err(),
        "{url} read anonymously — it is not private, so this test proves nothing"
    );

    // Then signed.
    let (_d2, dest2) = staging();
    let signed = S3Transport::new().signing_buckets([bucket]);
    let out = signed
        .fetch(&url, &dest2, &Conditional::default(), None)
        .expect("a named private bucket must read with the process's own credentials");
    assert!(out.bytes_written > 0, "signed read wrote nothing");
    assert_eq!(
        std::fs::metadata(&dest2).expect("staged file").len(),
        out.bytes_written,
        "the reported byte count must be the bytes on disk"
    );
    let etag = out.etag.clone().expect("S3 always returns an ETag");

    // And revalidation: the same ETag means NotModified with nothing downloaded.
    let (_d3, dest3) = staging();
    let again = signed
        .fetch(
            &url,
            &dest3,
            &Conditional {
                etag: Some(etag),
                last_modified: None,
            },
            None,
        )
        .expect("revalidation must not fail");
    assert_eq!(again.bytes_written, 0, "an unchanged object was re-downloaded");
}

/// **A bucket named in `EARTHSCI_S3_BUCKET_OPTIONS` reads signed, carrying
/// `x-amz-request-payer`.**
///
/// This is the shape a **requester-pays** bucket needs, and it is deliberately
/// asserted against a bucket that is *not* requester-pays yet. That is the whole
/// point: a requester-pays bucket refuses anonymous requests outright, so there
/// is no way to test the path against one without first breaking every anonymous
/// reader of it. S3 ignores the header on an ordinary bucket, so a signed read
/// carrying it proves the plumbing — the signature, the credential chain, the
/// header, the IAM grant — while the bucket is still public and nothing is at
/// risk. Flip the bucket afterwards and this same code is what runs.
///
/// Needs the network and AWS credentials that carry `s3:GetObject` on the
/// bucket, so it is opt-in through `EARTHSCI_S3_REQUESTER_PAYS_TEST=1`. Without
/// the IAM grant it fails as `403 AccessDenied`, which is the correct answer to
/// "may this identity read that bucket" and the thing worth finding out before a
/// deployment depends on it.
#[test]
fn a_named_bucket_reads_signed_as_a_requester() {
    let _lock = env_lock();
    if std::env::var("EARTHSCI_S3_REQUESTER_PAYS_TEST").is_err() {
        eprintln!(
            "set EARTHSCI_S3_REQUESTER_PAYS_TEST=1 plus AWS credentials with \
             s3:GetObject on inmap-model to run the live requester-pays read"
        );
        return;
    }
    // Named, so `signs()` says yes and the options reach `object_store`. The
    // region is stated here too, because a per-bucket region is the other half
    // of what one process-wide `EARTHSCI_S3_REGION` cannot express.
    std::env::set_var(
        earthsciio::BUCKET_OPTIONS_ENV,
        r#"{"inmap-model":{"aws_skip_signature":"false","aws_request_payer":"true","aws_region":"us-east-2"}}"#,
    );

    let (_dir, dest) = staging();
    let out = S3Transport::new()
        .fetch(
            "s3://inmap-model/isrm_v1.2.2.zarr/.zmetadata",
            &dest,
            &Conditional::default(),
            None,
        )
        .expect(
            "a signed requester-pays read must succeed — a 403 here is the IAM \
             grant missing, not the transport",
        );
    assert!(out.bytes_written > 0, "signed requester read wrote nothing");
    assert_eq!(
        std::fs::metadata(&dest).expect("staged file").len(),
        out.bytes_written
    );

    std::env::remove_var(earthsciio::BUCKET_OPTIONS_ENV);
}
