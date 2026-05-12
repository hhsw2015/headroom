// cbindgen is not used at compile time.
// To generate headroom.h manually:
//   cargo install cbindgen
//   cbindgen --lang C --crate headroom-ffi -o src/headroom.h
fn main() {
    // Make the dylib relocatable so consumers can locate it via @rpath
    // instead of hardcoding the build-time absolute path. On Linux we set
    // SONAME so ld.so respects -rpath / LD_LIBRARY_PATH lookups.
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target.as_str() {
        "macos" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libheadroom_ffi.dylib");
        }
        "linux" => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libheadroom_ffi.so");
        }
        _ => {}
    }
}
