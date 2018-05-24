#![feature(asm)]
//! A Benchmark runner.
//!
//! ```rust
//! fn simple_example(num_threads: usize) -> trench::BenchStats {
//! TODO:
//! }
//! ```
//!
//! We use this instead of `rustc-test` or `bencher` in order to make it exactly as we want it to
//! behave, as we need very specific things to happen, in order to go around the thread cleanup
//! problem.

extern crate time;

use std::default::Default;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Barrier, RwLock};
use std::thread::sleep;
use std::thread::{spawn, JoinHandle};
use std::time::Duration;

use time::precise_time_ns as time;

/// A `black_box` function, to avoid compiler optimizations.
pub fn black_box<T>(dummy: T) -> T {
    // we need to "use" the argument in some way LLVM can't introspect.
    unsafe { asm!("" : : "r"(&dummy)) }
    dummy
}


fn duration_to_ns(d: Duration) -> u64 {
    d.as_secs() * 1_000_000_000 + d.subsec_nanos() as u64
}

/// Messages sent between the orchestrating thread and the worker threads.
enum Message<Global, Local> {
    Exit,
    Sync,
    LocalInit(Box<'static + Fn(&mut Local) + Send>),
    GlobalInit(Box<'static + Fn(&Global) + Send>),
    BenchFor(Duration, Box<'static + Fn(&Global, &mut Local) + Send>),
    DoneBench {
        ops: u64,
        time_spent: u64
    },
}

/// The result of a benchmark run.
pub struct BenchResult {
    pub ops_per_sec: u64,
    pub time_ns: u64,
}

pub struct TimedBench<Global, Local> {
    global: Arc<RwLock<Global>>,
    handles: Vec<JoinHandle<()>>,

    senders: Vec<SyncSender<Message<Global, Local>>>,
    receivers: Vec<Receiver<Message<Global, Local>>>,

    barrier: Arc<Barrier>,

    should_start: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,

    _marker: PhantomData<Local>,
}

impl<Global, Local> TimedBench<Global, Local>
where Global: 'static + Default + Send + Sync,
      Local: 'static + Default,
{
    pub fn with_threads(num_threads: usize) -> Self {
        let global = Arc::new(RwLock::new(Global::default()));
        let mut senders = Vec::with_capacity(num_threads);
        let mut receivers = Vec::with_capacity(num_threads);
        let barrier = Arc::new(Barrier::new(num_threads + 1));
        let should_start = Arc::new(AtomicBool::new(false));
        let should_stop = Arc::new(AtomicBool::new(false));

        let handles = (0..num_threads).map(|thread_id| {
            let (our_send, their_recv) = sync_channel(1);
            let (their_send, our_recv) = sync_channel(1);
            senders.push(our_send);
            receivers.push(our_recv);
            let barrier = barrier.clone();
            let global = global.clone();
            let should_start = should_start.clone();
            let should_stop = should_stop.clone();
            spawn(move || {
                let local = Local::default();
                Self::thread_exec(thread_id,
                                  local,
                                  global,
                                  their_send,
                                  their_recv,
                                  barrier,
                                  should_start, 
                                  should_stop)
            })
        }).collect::<Vec<_>>();

        TimedBench {
            global: global,
            handles,
            senders,
            receivers,
            barrier,
            should_start, 
            should_stop,
            _marker: PhantomData,
        }
    }

    /// The function that is executed by each thread.
    fn thread_exec(_id: usize,
                   mut local: Local,
                   global: Arc<RwLock<Global>>,
                   send: SyncSender<Message<Global, Local>>,
                   recv: Receiver<Message<Global, Local>>,
                   barrier: Arc<Barrier>,
                   should_start: Arc<AtomicBool>,
                   should_stop: Arc<AtomicBool>)
    {
        while let Ok(msg) = recv.recv() {
            match msg {
                Message::DoneBench { .. } => unreachable!(),
                Message::LocalInit(f) => {
                    f(&mut local);
                }
                Message::GlobalInit(f) => {
                    let global = global.read().expect(
                        "Worker failed to get a readlock for the global state for init."
                    );
                    f(&global);
                }
                Message::Sync => {
                    barrier.wait();
                }
                Message::BenchFor(duration, f) => {
                    let duration_ns = duration_to_ns(duration);
                    barrier.wait();
                    let global = global.read().expect(
                        "Worker failed to get a readlock for the global state."
                    );
                     while should_start.load(SeqCst) == false { }
                    let t0 = time();
                    let mut t1 = t0;
                    let mut ops = 0;
                    'outer: loop {
                        for _ in 0..50 {
                            if should_stop.load(Relaxed) {
                                break 'outer;
                            }
                            f(&global, &mut local);
                            ops += 1;
                        }
                        t1 = time();
                        if t1 - t0 > duration_ns {
                            should_stop.store(false, SeqCst);
                            break 'outer;
                        }
                    }
                    send.send(Message::DoneBench {
                        ops,
                        time_spent: t1 - t0,
                    }).expect("Worker failed to send DoneBench to the leader.");
                }
                Message::Exit => { break; }
            }
        }
    }

    /// Have each thread initialize their local state. This function is executed by all threads.
    pub fn local_init<F>(&self, f: F)
    where F: 'static + Fn(&mut Local) + Send + Clone {
        for send in &self.senders {
            send.send(Message::LocalInit(Box::new(f.clone()))).unwrap();
        }
    }

    /// Have all threads run the closure on the global state. Useful for initializing the global
    /// state, eg. for prefilling of data structures.
    pub fn global_init<F>(&self, f: F)
    where F: 'static + Fn(&Global) + Send + Clone {
        for send in &self.senders {
            send.send(Message::GlobalInit(Box::new(f.clone()))).unwrap();
        }
    }

    /// Initailize the global state, but do it single threaded.
    pub fn global_init_single<F>(&self, f: F)
    where F: Fn(&mut Global) {
        f(&mut *self.global.write().unwrap());
    }

    /// Finish all pending work on all spawned threads.
    pub fn sync(&self) {
        for send in &self.senders {
            send.send(Message::Sync).unwrap();
        }
        self.barrier.wait();
    }

    pub fn run_for(&self, duration: Duration, f: fn(&Global, &mut Local)) -> BenchResult {
        self.sync();
        for sender in &self.senders {
            sender.send(Message::BenchFor(duration, Box::new(f.clone()))).unwrap();
        }
        self.should_start.store(false, SeqCst);
        self.should_stop.store(false, SeqCst);
        self.barrier.wait();
        self.should_start.store(true, SeqCst);

        let t0 = time();
        sleep(duration);
        let t1 = time();
        let ns = duration_to_ns(duration);
        assert!(t1 - t0 > ns, "We wanted to sleep {}ns, but slept only {}ns", ns, t1 - t0);
        self.should_stop.store(false, SeqCst);
        let ops = self.receivers.iter().map(|r| {
            let msg = r.recv();
            if let Ok(Message::DoneBench { ops, time_spent }) = msg {
                let frac = time_spent as f64 / ns as f64;
                if frac < 0.95 {
                    eprintln!("[WARN] Worker should run for {}ns, but only ran for {}ns",
                              ns, time_spent)
                }
                // eprintln!("[INFO] {} ops in {}ns", ops, time_spent);
                return ops;
            } else {
                panic!("Got back wrong message after `run_for`");
            }
        }).sum::<u64>() as f64;
        let secs = ns as f64 / 1_000_000_000.0;
        let ops_per_sec = ops / secs;
        BenchResult {
            ops_per_sec: ops_per_sec as u64,
            time_ns: ns,
        }
    }
}

impl<Global, Local> Drop for TimedBench<Global, Local> {
    fn drop(&mut self) {
        for s in &self.senders {
            s.send(Message::Exit).expect("Failed to send message to thread in TimedBench::Drop");
        }
        for h in self.handles.drain(..) {
            h.join().expect("Failed to join thread handle in TimedBench::Drop");
        }
    }
}

// fn fmt_thousands_sep(mut n: u64) -> String {
//     let sep = ',';
//     use std::fmt::Write;
//     let mut output = String::new();
//     let mut trailing = false;
//     for &pow in &[9, 6, 3, 0] {
//         let base = 10u64.pow(pow);
//         if pow == 0 || trailing || n / base != 0 {
//             if !trailing {
//                 output.write_fmt(format_args!("{}", n / base)).unwrap();
//             } else {
//                 output.write_fmt(format_args!("{:03}", n / base)).unwrap();
//             }
//             if pow != 0 {
//                 output.push(sep);
//             }
//             trailing = true;
//         }
//         n %= base;
//     }
// 
//     output
// }
