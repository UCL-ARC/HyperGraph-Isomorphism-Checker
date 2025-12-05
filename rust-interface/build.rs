fn main() {
    cc::Build::new()
        .file("process.c")
        .compile("process");
}
