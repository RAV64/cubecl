//! The stream-side graph-capture lifecycle, shared by every backend with
//! graph support (see [`ComputeServer::graph_prepare`](crate::server::ComputeServer::graph_prepare)).

use crate::metadata_cache::CacheMode;
use crate::server::ServerError;
use alloc::string::String;
use cubecl_environment::backtrace::BackTrace;

/// Where a stream sits in the graph-capture lifecycle. Capture is a strict
/// `NoCapture → Prepare → Capture → NoCapture` progression: `graph_prepare`
/// arms the pools (`NoCapture → Prepare`), `begin_capture` opens the recording
/// window (`Prepare → Capture`), and `end_capture` closes it (`Capture →
/// NoCapture`). Every transition rejects an out-of-order call, so a capture can
/// never start unprepared and two captures can never overlap on one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCaptureState {
    /// No capture is prepared or recording.
    NoCapture,
    /// `graph_prepare` has armed the persistent pools for the warmup run;
    /// `begin_capture` may now open the window. Slices the warmup run reserves
    /// are retained by the memory manager's priming (`CaptureState::primed`)
    /// until `begin_capture` calls `capture_priming_end`, so the pool ends up
    /// owning the capture run's full working set.
    Prepare,
    /// Launches are being recorded into a graph instead of executing. On a
    /// hardware-graph backend (CUDA, HIP) a host sync issued now aborts the
    /// driver capture, so the execution path defers fenced flushes until
    /// `end_capture`; a software-graph backend rejects the offending call
    /// directly.
    Capture,
}

impl StreamCaptureState {
    /// Whether launches on the stream are being recorded into a graph right
    /// now — the window during which a host sync would abort (or is rejected
    /// by) the capture.
    pub fn is_recording(&self) -> bool {
        matches!(self, StreamCaptureState::Capture)
    }

    /// The [`CacheMode`] the metadata info cache should run in at this lifecycle
    /// position. Both while a graph is being *prepared* (warmup, which primes
    /// the cache) and while it is being *recorded* the cache runs in
    /// [`CacheMode::Capture`] — caching every buffer and invalidating none — so
    /// the capture window finds every info buffer warm and drops none out from
    /// under a recorded launch. Normal operation uses [`CacheMode::Normal`].
    pub fn cache_mode(&self) -> CacheMode {
        match self {
            StreamCaptureState::NoCapture => CacheMode::Normal,
            StreamCaptureState::Prepare | StreamCaptureState::Capture => CacheMode::Capture,
        }
    }
}

/// Build a [`ServerError`] for a graph-capture call issued in the wrong state
/// (e.g. `begin_capture` without `graph_prepare`, or a second overlapping
/// capture on the same stream).
pub fn graph_state_error(reason: impl Into<String>) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}
