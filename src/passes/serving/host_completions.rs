//! Stream-ordered device host completion callback queue and background dispatcher.
//!
//! Decouples device-stream host callbacks from runtime locks by recording `(stream_ptr, kind,
//! payload)` completions on a lock-free queue (`record_completion_on_stream`) and draining them
//! on a dispatcher worker (`drain_once`) into registered per-kind handlers with dispatch and
//! exception counters.

use indexmap::IndexMap;
use std::collections::VecDeque;
use std::fmt;

/// A single stream-ordered completion callback item recorded on a device stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingStreamCompletion {
    /// Device stream handle on which the callback was ordered.
    pub stream_ptr: u64,
    /// Registered completion kind (e.g. `"finish_write"`, `"finish_read_prefetched"`).
    pub kind: String,
    /// Encoded payload bytes delivered to the handler.
    pub payload: Vec<u8>,
}

type HostCompletionHandler = Box<dyn FnMut(&[u8]) -> Result<(), String> + Send>;

/// Drains buffered device-stream completions and dispatches each payload to the handler registered
/// for its `kind` without holding runtime locks on the device driver thread.
pub struct DeviceHostFuncDispatcher {
    handlers: IndexMap<String, HostCompletionHandler>,
    pending_queue: VecDeque<PendingStreamCompletion>,
    dispatched_count: usize,
    dropped_unregistered_count: usize,
    exception_counts: IndexMap<String, usize>,
    running: bool,
}

impl fmt::Debug for DeviceHostFuncDispatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHostFuncDispatcher")
            .field(
                "registered_kinds",
                &self.handlers.keys().collect::<Vec<_>>(),
            )
            .field("pending_count", &self.pending_queue.len())
            .field("dispatched_count", &self.dispatched_count)
            .field(
                "dropped_unregistered_count",
                &self.dropped_unregistered_count,
            )
            .field("exception_counts", &self.exception_counts)
            .field("running", &self.running)
            .finish()
    }
}

impl Default for DeviceHostFuncDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceHostFuncDispatcher {
    /// Creates a new [`DeviceHostFuncDispatcher`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            handlers: IndexMap::new(),
            pending_queue: VecDeque::new(),
            dispatched_count: 0,
            dropped_unregistered_count: 0,
            exception_counts: IndexMap::new(),
            running: true,
        }
    }

    /// Registers a completion handler for `kind`.
    pub fn register<F>(&mut self, kind: impl Into<String>, handler: F)
    where
        F: FnMut(&[u8]) -> Result<(), String> + Send + 'static,
    {
        self.handlers.insert(kind.into(), Box::new(handler));
    }

    /// Schedules a stream-ordered host completion for `kind` with `payload` on `stream_ptr`.
    pub fn record_completion_on_stream(
        &mut self,
        stream_ptr: u64,
        kind: impl Into<String>,
        payload: Vec<u8>,
    ) {
        self.pending_queue.push_back(PendingStreamCompletion {
            stream_ptr,
            kind: kind.into(),
            payload,
        });
    }

    /// Drains all currently queued completions and dispatches each to its registered handler.
    ///
    /// Handler errors are caught and counted in `handler_exception_counts` so the drain loop stays
    /// alive across individual handler failures. Returns the number of successfully dispatched
    /// callbacks in this drain cycle.
    pub fn drain_once(&mut self) -> usize {
        let before = self.dispatched_count;
        let completions: Vec<PendingStreamCompletion> = self.pending_queue.drain(..).collect();
        for item in completions {
            let Some(handler) = self.handlers.get_mut(&item.kind) else {
                self.dropped_unregistered_count += 1;
                continue;
            };
            match handler(&item.payload) {
                Ok(()) => {
                    self.dispatched_count += 1;
                }
                Err(_) => {
                    *self.exception_counts.entry(item.kind).or_insert(0) += 1;
                }
            }
        }
        self.dispatched_count - before
    }

    /// Stops the dispatcher and executes one final drain of any remaining completions.
    pub fn stop(&mut self) -> usize {
        self.running = false;
        self.drain_once()
    }

    /// Returns the total number of successfully dispatched completions.
    #[must_use]
    pub const fn dispatched_count(&self) -> usize {
        self.dispatched_count
    }

    /// Returns the number of completions dropped because no handler was registered for their
    /// `kind`.
    #[must_use]
    pub const fn dropped_unregistered_count(&self) -> usize {
        self.dropped_unregistered_count
    }

    /// Returns the per-kind handler exception counts.
    #[must_use]
    pub fn handler_exception_counts(&self) -> IndexMap<String, usize> {
        self.exception_counts.clone()
    }

    /// Returns the number of buffered completions awaiting drain.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    /// Returns whether the dispatcher is currently marked running.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_device_host_func_dispatcher_drains_and_counts_exceptions() {
        let mut disp = DeviceHostFuncDispatcher::new();
        let seen: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_clone = Arc::clone(&seen);

        disp.register("finish_write", move |payload| {
            if payload.is_empty() {
                return Err("empty payload".to_string());
            }
            seen_clone.lock().expect("lock").extend_from_slice(payload);
            Ok(())
        });

        disp.record_completion_on_stream(0x10, "finish_write", vec![1, 2, 3]);
        disp.record_completion_on_stream(0x10, "finish_write", Vec::new());
        disp.record_completion_on_stream(0x10, "unknown_kind", vec![9]);

        let ok_count = disp.drain_once();
        assert_eq!(ok_count, 1);
        assert_eq!(disp.dispatched_count(), 1);
        assert_eq!(disp.dropped_unregistered_count(), 1);
        assert_eq!(
            disp.handler_exception_counts().get("finish_write"),
            Some(&1)
        );
        assert_eq!(*seen.lock().expect("lock"), vec![1, 2, 3]);
    }
}
