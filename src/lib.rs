#![feature(asm)]
#![allow(dead_code)]
/// A Benchmark runner.
///
/// We use this instead of `rustc-test` or `bencher` in order to make it exactly as we want it to
/// behave, as we need very specific things to happen, in order to go around the thread cleanup
/// problem.

extern crate time;

use std::sync::mpsc::{Sender, Receiver, channel};
use std::sync::{Arc, Barrier};
use std::thread;

const DEFAULT_NUM_SAMPLES: usize = 200;

/// Statistics from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchStats {
    /// An identifier for what benchmark this is.
    ident: String,
    /// The timings from this benchmark.
    samples: Vec<u64>,
}

impl BenchStats {
    pub fn ident(&self) -> &str {
        &self.ident
    }
}

impl BenchStats {
    fn len(&self) -> u64 {
        self.samples.len() as u64
    }

    /// Get the average of the samples
    pub fn average(&self) -> u64 {
        self.samples.iter().cloned().sum::<u64>() / self.len()
    }

    /// Get the variance of the samples
    pub fn variance(&self) -> u64 {
        let avg = self.average();
        let s = self.samples
            .iter()
            .cloned()
            .map(|s| (if s < avg { (avg - s) } else { s - avg }).pow(2))
            .sum::<u64>() / self.len();
        (s as f32).sqrt() as u64
    }

    /// Get the minimum of the samples
    pub fn min(&self) -> u64 {
        self.samples.iter().cloned().min().unwrap()
    }

    /// Get the maximum of the samples
    pub fn max(&self) -> u64 {
        self.samples.iter().cloned().max().unwrap()
    }

    /// Get the number of samples that are above the average
    pub fn above_avg(&self) -> u64 {
        let avg = self.average();
        self.samples.iter().filter(|&&s| s > avg).count() as u64
    }

    /// Get the number of samples that are below the average
    pub fn below_avg(&self) -> u64 {
        let avg = self.average();
        self.samples.iter().filter(|&&s| s < avg).count() as u64
    }

    /// Get a human readable string of these statistics. This is indended to mimic the format used
    /// by `cargo bench`.
    pub fn report(&self) -> String {
        format!(
            "{} ns/iter (+/- {}) min={} max={} above={} below={}",
            Self::fmt_thousands_sep(self.average()),
            Self::fmt_thousands_sep(self.variance()),
            self.min(),
            self.max(),
            self.above_avg(),
            self.below_avg()
        )
    }

    /// Write out the statistics separated by `;`. This does not write out the samples; it is a
    /// single line with average, variance, etc.
    pub fn csv(&self) -> String {
        format!(
            "{};{};{};{};{};{}",
            Self::fmt_thousands_sep(self.average()),
            Self::fmt_thousands_sep(self.variance()),
            self.min(),
            self.max(),
            self.above_avg(),
            self.below_avg()
        )
    }

    /// Get the csv header.
    pub fn csv_header() -> String {
        format!(
            "{};{};{};{};{};{}",
            "average",
            "variance",
            "min",
            "max",
            "# above avg",
            "# below avg"
        )
    }

    /// Get the samples.
    pub fn samples(&self) -> &[u64] {
        &self.samples
    }

    // This is borrowed from `test::Bencher` :)
    fn fmt_thousands_sep(mut n: u64) -> String {
        let sep = ',';
        use std::fmt::Write;
        let mut output = String::new();
        let mut trailing = false;
        for &pow in &[9, 6, 3, 0] {
            let base = 10u64.pow(pow);
            if pow == 0 || trailing || n / base != 0 {
                if !trailing {
                    output.write_fmt(format_args!("{}", n / base)).unwrap();
                } else {
                    output.write_fmt(format_args!("{:03}", n / base)).unwrap();
                }
                if pow != 0 {
                    output.push(sep);
                }
                trailing = true;
            }
            n %= base;
        }

        output
    }
}

/// Turn the statistics given into a gnuplot data string.
pub fn gnuplot(stats: &[BenchStats]) -> String {
    let mut s = String::new();
    let lines = stats.iter().map(|b| b.samples.len()).max().unwrap_or(0);
    for stats in stats {
        s.push_str(&stats.ident());
    }
    s.push('\n');
    for i in 0..lines {
        for stat in stats {
            s.push_str(&format!("{} ", stat.samples.get(i).cloned().unwrap_or(0)));
        }
        s.push('\n');
    }

    s
}

/// A `black_box` function, to avoid compiler optimizations.
pub fn black_box<T>(dummy: T) -> T {
    // we need to "use" the argument in some way LLVM can't introspect.
    unsafe { asm!("" : : "r"(&dummy)) }
    dummy
}

pub trait Spawner {
    type Return;
    type Result;

    fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> Self::Return,
        F: Send + 'static,
        Self::Return: Send + 'static;

    fn join(self) -> Self::Result;
}

/// A `Spawner` from the standard library.
pub struct StdThread<T> {
    handle: std::thread::JoinHandle<T>,
}

#[derive(Debug)]
struct FunctionPtr<State> {
    data: usize,
    state: *const State,
    _marker: ::std::marker::PhantomData<State>,
}

impl<S> Clone for FunctionPtr<S> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            state: self.state,
            _marker: ::std::marker::PhantomData,
        }
    }
}

impl<S> FunctionPtr<S> {
    fn new(f: fn(&S), state: &S) -> Self {
        FunctionPtr {
            data: f as usize,
            state: &*state,
            _marker: ::std::marker::PhantomData,
        }
    }

    fn call(&mut self) {
        unsafe {
            let f = ::std::mem::transmute::<usize, fn(&S)>(self.data);
            (f)(&*self.state);
        }
    }
}

unsafe impl<S> Send for FunctionPtr<S> {}

#[derive(Debug)]
enum ThreadSignal<S> {
    Run(FunctionPtr<S>),
    Done(u64),
    End,
}

impl<T> Spawner for StdThread<T>
where
    T: Send,
{
    type Return = T;
    type Result = thread::Result<T>;

    fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> Self::Return,
        F: Send + 'static,
        Self::Return: Send + 'static,
    {
        StdThread { handle: thread::spawn(f) }
    }

    fn join(self) -> Self::Result {
        self.handle.join()
    }
}

/// A `Bencher` is the struct the user uses to make a benchmark. The `State` type is a user chosen
/// type that the benchmarking function is given a reference to.
pub struct Bencher<State, Sp: Spawner = StdThread<()>> {
    /// Samples for this benchmark
    samples: Vec<u64>,
    /// The state.
    state: State,
    /// The number of runs
    n: usize,
    /// A vector of thread handles.
    threads: Vec<Sp>,
    senders: Vec<Sender<ThreadSignal<State>>>,
    receivers: Vec<Receiver<ThreadSignal<State>>>,
    /// A function to be executed before each benchmark run. This is only ran by one thread.
    before: Box<Fn(&mut State)>,
    /// A function to be executed after each benchmark run. This is only ran by one thread.
    after: Box<Fn(&mut State)>,
    barrier: Arc<Barrier>,
}

impl<St, Sp> Bencher<St, Sp>
where
    St: 'static,
    Sp: Spawner,
    Sp::Return: Send + Default + 'static,
{
    /// Construct a new `Bencher`. The initial state and the number of threads is given.
    pub fn new(state: St, n_threads: usize) -> Self {
        let mut senders = Vec::with_capacity(n_threads);
        let mut receivers = Vec::with_capacity(n_threads);
        let barrier = Arc::new(Barrier::new(n_threads + 1));
        // Start the threads, and give them channels for communication.
        let threads = (0..n_threads)
            .map(|_thread_id| {
                let (our_send, their_recv) = channel();
                let (their_send, our_recv) = channel();
                senders.push(our_send);
                receivers.push(our_recv);
                let barrier = barrier.clone();
                Sp::spawn(move || {
                    let recv = their_recv;
                    let send = their_send;
                    loop {
                        let signal = match recv.recv() {
                            Ok(ThreadSignal::Run(ref mut f)) => {
                                barrier.wait();
                                let t0 = time::precise_time_ns();
                                f.call();
                                let t1 = time::precise_time_ns();
                                ThreadSignal::Done(t1 - t0)
                            }
                            Ok(ThreadSignal::End) => {
                                break;
                            }
                            Ok(_) => unreachable!(),
                            Err(e) => panic!("{:?}", e),
                        };
                        assert!(send.send(signal).is_ok());
                    }
                    Default::default()
                })
            })
            .collect();
        Self {
            state,
            samples: vec![],
            n: DEFAULT_NUM_SAMPLES,
            threads,
            senders,
            receivers,
            before: Box::new(|_| {}),
            after: Box::new(|_| {}),
            barrier,
        }
    }

    /// Start a threaded benchmark. All threads will run the function given. The state passed in is
    /// shared between all threads.
    pub fn bench(&mut self, f: fn(&St)) {
        let func_ptr = FunctionPtr::new(f, &self.state);
        for _i in 0..self.n {
            (self.before)(&mut self.state);
            for sender in &self.senders {
                assert!(sender.send(ThreadSignal::Run(func_ptr.clone())).is_ok());
            }
            // TODO: this is not good: we risk waiting for a long time in `barrier.wait`
            let t0 = time::precise_time_ns();
            self.barrier.wait();
            for recv in self.receivers.iter() {
                match recv.recv() {
                    Ok(ThreadSignal::Done(_t)) => {
                        // OK
                    }
                    _ => panic!("Thread didn't return correctly"),
                }
            }
            let t1 = time::precise_time_ns();
            self.samples.push(t1 - t0);
        }
        for sender in &self.senders {
            assert!(sender.send(ThreadSignal::End).is_ok());
        }
        (self.after)(&mut self.state);
    }

    /// Set the closure that is ran before each benchmark iteration.
    pub fn before<F: 'static + Fn(&mut St)>(&mut self, f: F) {
        self.before = Box::new(f);
    }

    /// Set the closure that is ran afetr each benchmark iteration.
    pub fn after<F: 'static + Fn(&mut St)>(&mut self, f: F) {
        self.after = Box::new(f);
    }

    /// Convert the Becnher into `BenchStats` for statistics reporting.
    pub fn into_stats<S>(self, name: S) -> BenchStats
    where
        S: Into<String>,
    {
        self.threads.into_iter().map(Spawner::join).count();
        BenchStats {
            samples: self.samples,
            ident: name.into(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        #[derive(Debug, Default, Clone)]
        struct State;

        #[inline(never)]
        fn sample_function(_state: &State) {
            let mut s = 0;
            for i in 0..12345 {
                s += i;
            }
            black_box(s);
        }

        let mut b = Bencher::<State>::new(State, 4);
        b.bench(sample_function);
        let st = b.into_stats("test benchmark");
        println!("{}", st.report());
    }
}
