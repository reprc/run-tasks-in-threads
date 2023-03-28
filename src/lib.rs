#![feature(int_roundings)]

use itertools::Itertools;
use std::{
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    vec,
};

fn compute<T, R>(data: Vec<T>, func: fn(T) -> R, threshold: usize) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    match n_batches(data.len(), threshold) {
        0 => {
            vec![]
        }
        1 => data.into_iter().map(func).collect(),
        _ => {
            let batches = split_data_into_batches(data, threshold);
            process_batches_in_threads(batches, func)
        }
    }
}

fn n_batches(data_size: usize, batch_size: usize) -> usize {
    data_size.div_ceil(batch_size)
}

fn split_data_into_batches<T>(data: Vec<T>, batch_size: usize) -> Vec<Vec<T>> {
    data.into_iter()
        .chunks(batch_size)
        .into_iter()
        .map(|chunk| chunk.collect_vec())
        .collect()
}

fn process_batches_in_threads<T, R>(batches: Vec<Vec<T>>, func: fn(T) -> R) -> Vec<R>
where
    T: Send + 'static,
    R: Send + 'static,
{
    let max_threads = num_cpus();
    let mut handles = Vec::with_capacity(batches.len());

    let running_threads = Arc::new(Mutex::new(0));
    let cvar = Arc::new(Condvar::new());

    for batch in batches {
        *running_threads.lock().unwrap() += 1;

        let running_threads_clone = running_threads.clone();
        let cvar_clone = cvar.clone();

        // if thread panics, number of running threads won't be reduced
        let handle = thread::spawn(move || {
            let result = batch.into_iter().map(func).collect();
            *running_threads_clone.lock().unwrap() -= 1;
            cvar_clone.notify_all();
            result
        });
        handles.push(handle);

        let _guard = cvar
            .wait_while(running_threads.lock().unwrap(), |running_threads| {
                *running_threads == max_threads
            })
            .unwrap();
    }
    aggregate_results(handles)
}

fn aggregate_results<T>(handles: Vec<JoinHandle<Vec<T>>>) -> Vec<T> {
    let mut result = vec![];
    for handle in handles {
        let thread_id = handle.thread().id();
        match handle.join() {
            Ok(thread_result) => result.extend(thread_result),
            Err(_) => {
                eprintln!("thread {:?} panicked", thread_id)
            }
        }
    }
    result
}

fn num_cpus() -> usize {
    4
}

#[cfg(test)]
mod tests {
    use crate::{compute, n_batches, split_data_into_batches};

    #[test]
    fn test_n_batches() {
        assert_eq!(n_batches(0, 2), 0);
        assert_eq!(n_batches(2, 2), 1);
        assert_eq!(n_batches(3, 2), 2);
    }

    #[test]
    fn test_split_into_batches() {
        let data = vec![1, 2, 3, 4, 5];
        let expected = vec![vec![1, 2], vec![3, 4], vec![5]];
        let split = split_data_into_batches(data, 2);
        assert_eq!(expected, split);
    }

    fn func(n: u32) -> u64 {
        if n == 42 {
            panic!()
        }
        (n + 42) as u64
    }

    #[test]
    fn test_compute_empty() {
        let result = compute(vec![], func, 2);
        assert!(result.is_empty())
    }

    #[test]
    fn test_compute_single_batch() {
        let expected = vec![42, 142];
        let data = vec![0, 100];
        let result = compute(data, func, 2);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_compute_multiple_batch() {
        let expected = vec![42, 142, 192, 242, 292];
        let data = vec![0, 100, 150, 200, 250];
        let result = compute(data, func, 2);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_compute_multiple_batch_with_panic() {
        // func panics so whole batch expected to be discarded
        let expected = vec![192, 242, 292];
        let data = vec![42, 100, 150, 200, 250];
        let result = compute(data, func, 2);
        assert_eq!(expected, result);
    }
}
