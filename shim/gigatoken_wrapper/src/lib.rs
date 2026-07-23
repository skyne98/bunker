// Pre-fork N tokenizers at init, hand out via atomic counter. Zero fork overhead at encode time.
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use gigatoken_rs::bpe::tiktoken::Tokenizer;
use gigatoken_rs::pretokenize::PretokenizerType;
use rayon::prelude::*;

pub struct GtTokenizer { pool: Vec<Mutex<Tokenizer>> }

#[unsafe(no_mangle)]
pub extern "C" fn gt_init(path: *const std::os::raw::c_char) -> *mut GtTokenizer {
    if path.is_null() { return std::ptr::null_mut(); }
    let s = match unsafe { CStr::from_ptr(path) }.to_str() { Ok(s) => s, Err(_) => return std::ptr::null_mut() };
    match load_hf_bpe(&PathBuf::from(s)) {
        Ok(mut t) => {
            t.set_pretokenizer_type(PretokenizerType::Qwen35);
            let n = num_cpus();
            let pool: Vec<Mutex<Tokenizer>> = (0..n).map(|_| Mutex::new(t.fork())).collect();
            Box::into_raw(Box::new(GtTokenizer { pool }))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(32)
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_encode(tok: *mut GtTokenizer, text: *const u8, text_len: usize, out_buf: *mut u32, out_max: usize) -> std::os::raw::c_int {
    if tok.is_null() || text.is_null() || out_buf.is_null() { return -1; }
    let gt = unsafe { &*tok };
    let bytes = unsafe { std::slice::from_raw_parts(text, text_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf, out_max) };
    let mut out: Vec<u32> = Vec::with_capacity(text_len);
    gt.pool[0].lock().unwrap().encode_with_added_tokens_flat(bytes, &mut out);
    let n = out.len().min(out_max);
    out_slice[..n].copy_from_slice(&out[..n]);
    n as std::os::raw::c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_encode_mt(tok: *mut GtTokenizer, text: *const u8, text_len: usize, out_buf: *mut u32, out_max: usize) -> std::os::raw::c_int {
    if tok.is_null() || text.is_null() || out_buf.is_null() { return -1; }
    let gt = unsafe { &*tok };
    let bytes = unsafe { std::slice::from_raw_parts(text, text_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf, out_max) };

    const CHUNK: usize = 4 * 1024 * 1024;
    let chunks: Vec<&[u8]> = if text_len <= CHUNK {
        vec![bytes]
    } else {
        let mut v = Vec::new();
        let mut s = 0;
        while s < text_len {
            let mut e = (s + CHUNK).min(text_len);
            if e < text_len { while e < text_len && bytes[e] != b' ' { e += 1; } if e < text_len { e += 1; } }
            v.push(&bytes[s..e]); s = e;
        }
        v
    };

    let counter = AtomicUsize::new(0);
    let pool = &gt.pool;
    let results: Vec<Vec<u32>> = chunks.par_iter().map(|chunk| {
        let idx = counter.fetch_add(1, Ordering::Relaxed) % pool.len();
        let mut out = Vec::with_capacity(chunk.len());
        pool[idx].lock().unwrap().encode_with_added_tokens_flat(chunk, &mut out);
        out
    }).collect();

    let mut total = 0usize;
    for ids in &results {
        let n = ids.len().min(out_max - total);
        out_slice[total..total + n].copy_from_slice(&ids[..n]);
        total += n;
    }
    total as std::os::raw::c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_free(tok: *mut GtTokenizer) {
    if !tok.is_null() { unsafe { drop(Box::from_raw(tok)) }; }
}
