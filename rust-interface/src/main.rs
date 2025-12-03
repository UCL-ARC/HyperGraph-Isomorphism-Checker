#[repr(C)]
pub struct Slice {
    ptr: *const f64,
    len: usize,
}

extern "C" {
    fn process_array(slice: Slice);
}

fn main() {
    let data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

    let slice = Slice {
        ptr: data.as_ptr(),
        len: data.len(),
    };

    unsafe {
        process_array(slice);
    }
}
