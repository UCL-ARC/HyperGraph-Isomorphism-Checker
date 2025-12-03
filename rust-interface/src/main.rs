#[repr(C)]
pub struct Slice {
    ptr: *const u64,
    len: usize,
}

extern "C" {
    fn process_array(slice: Slice);
}

fn main() {
    let data: [u64; 5] = [1, 2, 3, 4, 5];

    let slice = Slice {
        ptr: data.as_ptr(),
        len: data.len(),
    };

    unsafe {
        process_array(slice);
    }
}
