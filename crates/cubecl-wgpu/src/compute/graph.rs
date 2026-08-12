use crate::WgpuResource;
use crate::schedule::Addresses;
use cubecl_runtime::memory_management::{ManagedMemoryHandle, SharedMemoryBindings};
use std::sync::Arc;
use wgpu::ComputePipeline;

/// A captured wgpu graph: the recorded launch sequence, fully resolved.
///
/// WebGPU has no driver-side graph object and no re-submittable command
/// buffers (`queue.submit` consumes them), so unlike CUDA/HIP the graph is a
/// **software graph**: everything a launch resolves per dispatch — pipeline
/// lookup, binding resolution, info-uniform upload, bind group creation — is
/// done once while recording, and [`WgpuStream::replay_graph`](super::stream::WgpuStream::replay_graph)
/// re-encodes the prebuilt state in a tight loop. Replay cost stays O(n) in
/// recorded tasks (encoding cannot be skipped under WebGPU), but with a far
/// smaller constant than the full launch path.
///
/// Owned by the [`WgpuServer`](super::server::WgpuServer) registry and
/// referenced by [`GraphId`](cubecl_runtime::id::GraphId); the client
/// references the graph by id and, on the last drop, asks the server to
/// release it.
#[derive(Debug)]
pub struct WgpuGraph {
    /// The recorded tasks, replayed in order.
    pub(crate) tasks: Vec<ReplayTask>,
    /// Every pool slice the capture window allocated (intermediates, info
    /// uniforms, Vulkan address buffers), pinned for the graph's lifetime. A
    /// replay re-runs the recorded dispatches against these exact buffers;
    /// retaining the handles keeps the memory pools from reusing the slices.
    /// Dropped with the graph, releasing the memory.
    pub(crate) _retained: Vec<ManagedMemoryHandle>,
    /// Cross-stream input bindings the recorded tasks reference, pinned for
    /// the graph's lifetime instead of until the next submission (the normal
    /// path's release point, see [`WgpuStream::flush`](super::stream::WgpuStream)).
    pub(crate) _shared: SharedMemoryBindings,
}

/// One recorded dispatch, resolved down to what `wgpu` needs at encode time.
#[derive(Debug)]
pub(crate) struct ReplayTask {
    pub(crate) pipeline: Arc<ComputePipeline>,
    /// Built once at record time; `wgpu` bind groups are reusable, and the
    /// buffers they reference are pinned by the graph, so the group stays
    /// valid for the graph's lifetime. `None` when the kernel binds no
    /// resources (Vulkan immediate-address mode).
    pub(crate) bind_group: Option<wgpu::BindGroup>,
    /// Vulkan buffer device addresses passed as immediates. Stable across
    /// replays because the buffers they point into are pinned.
    pub(crate) immediates: Option<Addresses>,
    /// Buffers needing an explicit transition to storage read-write state
    /// (Vulkan buffer-address mode, where usage tracking cannot see them).
    pub(crate) transitions: Vec<WgpuResource>,
    pub(crate) dispatch: ReplayDispatch,
}

/// The dispatch shape of a recorded task.
#[derive(Debug)]
pub(crate) enum ReplayDispatch {
    Static(u32, u32, u32),
    /// Indirect dispatch: the workgroup count is read from this (pinned)
    /// buffer at execution time, so replays pick up updated counts written
    /// between them.
    Dynamic(WgpuResource),
}

/// The in-progress recording on a stream, moved into a [`WgpuGraph`] at
/// `end_capture`.
#[derive(Debug, Default)]
pub(crate) struct GraphRecording {
    pub(crate) tasks: Vec<ReplayTask>,
    /// Cross-stream input bindings of recorded tasks (see
    /// [`WgpuGraph::_shared`]).
    pub(crate) shared: SharedMemoryBindings,
    /// Uniform slices created inside the recording window (info uniforms on a
    /// cache miss, Vulkan address buffers). Holding the handles keeps the
    /// slices alive until `end_capture`, where the memory manager's
    /// `capture_end` retains every live touched slice on the graph — without
    /// this, the retention would silently depend on no uniform release running
    /// mid-window.
    pub(crate) uniform_pins: Vec<ManagedMemoryHandle>,
}
