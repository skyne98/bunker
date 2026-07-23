// Best version: atomic write cursor + thread_local Vec + UnsafeCell. No Mutex, no clone, no compaction.
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::{RefCell, UnsafeCell};
use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use gigatoken_rs::bpe::tiktoken::Tokenizer;
use gigatoken_rs::pretokenize::PretokenizerType;
use rayon::prelude::*;

struct UT(UnsafeCell<Tokenizer>);
unsafe impl Send for UT {}
unsafe impl Sync for UT {}

pub struct GtTokenizer { pool: Vec<UT> }
thread_local! { static LB: RefCell<Vec<u32>> = RefCell::new(Vec::new()); }

#[unsafe(no_mangle)]
pub extern "C" fn gt_init(path: *const std::os::raw::c_char) -> *mut GtTokenizer {
    if path.is_null() { return std::ptr::null_mut(); }
    let s = match unsafe { CStr::from_ptr(path) }.to_str() { Ok(s) => s, Err(_) => return std::ptr::null_mut() };
    match load_hf_bpe(&PathBuf::from(s)) {
        Ok(mut t) => {
            t.set_pretokenizer_type(PretokenizerType::Qwen35);
            let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(32);
            let pool: Vec<UT> = (0..n).map(|_| UT(UnsafeCell::new(t.fork()))).collect();
            Box::into_raw(Box::new(GtTokenizer { pool }))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_encode(tok: *mut GtTokenizer, text: *const u8, text_len: usize, out_buf: *mut u32, out_max: usize) -> std::os::raw::c_int {
    if tok.is_null() || text.is_null() || out_buf.is_null() { return -1; }
    let gt = unsafe { &*tok };
    let bytes = unsafe { std::slice::from_raw_parts(text, text_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf, out_max) };
    let mut out: Vec<u32> = Vec::with_capacity(text_len);
    unsafe { (&mut *gt.pool[0].0.get()).encode_with_added_tokens_flat(bytes, &mut out); }
    let n = out.len().min(out_max);
    out_slice[..n].copy_from_slice(&out[..n]);
    n as std::os::raw::c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_encode_mt(tok: *mut GtTokenizer, text: *const u8, text_len: usize, out_buf: *mut u32, out_max: usize) -> std::os::raw::c_int {
    if tok.is_null() || text.is_null() || out_buf.is_null() { return -1; }
    let gt = unsafe { &*tok };
    let bytes = unsafe { std::slice::from_raw_parts(text, text_len) };
    let n_threads = gt.pool.len();
    let pool_ptr = gt.pool.as_ptr() as usize;
    let out_ptr = out_buf as usize;

    // Split into n_threads chunks at space boundaries
    let chunk_size = (text_len + n_threads - 1) / n_threads;
    let mut chunks: Vec<&[u8]> = Vec::new();
    let mut s = 0;
    while s < text_len {
        let mut e = (s + chunk_size).min(text_len);
        if e < text_len { while e < text_len && bytes[e] != b' ' { e += 1; } if e < text_len { e += 1; } }
        chunks.push(&bytes[s..e]); s = e;
    }

    let counter = AtomicUsize::new(0);
    let write_cursor = AtomicUsize::new(0);

    chunks.par_iter().for_each(|chunk| {
        let idx = counter.fetch_add(1, Ordering::Relaxed) % n_threads;
        let tok = unsafe { &mut *((pool_ptr as *mut UT).add(idx)) };
        let tok = unsafe { &mut *tok.0.get() };

        LB.with(|buf| {
            let mut buf = buf.borrow_mut();
            buf.clear();
            buf.reserve(chunk.len());
            tok.encode_with_added_tokens_flat(chunk, &mut buf);

            let offset = write_cursor.fetch_add(buf.len(), Ordering::Relaxed);
            if offset + buf.len() <= out_max {
                let dst = unsafe { std::slice::from_raw_parts_mut((out_ptr as *mut u32).add(offset), buf.len()) };
                dst.copy_from_slice(&buf[..]);
            }
        });
    });

    write_cursor.load(Ordering::Relaxed) as std::os::raw::c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_free(tok: *mut GtTokenizer) {
    if !tok.is_null() { unsafe { drop(Box::from_raw(tok)) }; }
}
