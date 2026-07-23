// C ABI wrapper for gigatoken — zero-copy, caller-allocated output buffer.
use std::ffi::CStr;
use std::path::PathBuf;
use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;
use gigatoken_rs::bpe::tiktoken::Tokenizer;
use gigatoken_rs::pretokenize::PretokenizerType;

pub struct GtTokenizer { inner: Tokenizer }

#[unsafe(no_mangle)]
pub extern "C" fn gt_init(path: *const std::os::raw::c_char) -> *mut GtTokenizer {
    if path.is_null() { return std::ptr::null_mut(); }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() { Ok(s) => s, Err(_) => return std::ptr::null_mut() };
    match load_hf_bpe(&PathBuf::from(path_str)) {
        Ok(mut tok) => { tok.set_pretokenizer_type(PretokenizerType::Qwen35); Box::into_raw(Box::new(GtTokenizer { inner: tok })) }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Encode text into a caller-provided u32 buffer. Returns the number of tokens written,
/// or -1 on error. The buffer must be large enough (text_len is a safe upper bound).
#[unsafe(no_mangle)]
pub extern "C" fn gt_encode(
    tok: *mut GtTokenizer,
    text: *const u8, text_len: usize,
    out_buf: *mut u32, out_max: usize,
) -> std::os::raw::c_int {
    if tok.is_null() || text.is_null() || out_buf.is_null() { return -1; }
    let gt = unsafe { &mut *tok };
    let bytes = unsafe { std::slice::from_raw_parts(text, text_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf, out_max) };
    let mut out: Vec<u32> = Vec::with_capacity(text_len);
    gt.inner.encode_with_added_tokens_flat(bytes, &mut out);
    let n = out.len().min(out_max);
    out_slice[..n].copy_from_slice(&out[..n]);
    n as std::os::raw::c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_free(tok: *mut GtTokenizer) {
    if !tok.is_null() { unsafe { drop(Box::from_raw(tok)) }; }
}
