// SPDX-License-Identifier: GPL-2.0 OR MIT

//! DRM framebuffers.
//!
//! C header: [`include/drm/drm_framebuffer.h`](srctree/include/drm/drm_framebuffer.h)

use super::{KmsDriver, ModeObject, Sealed};
use crate::{
    drm::device::Device,
    prelude::*,
    sync::aref::{ARef, AlwaysRefCounted},
    types::*,
};
#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
use crate::{
    drm::gem::{self, shmem, BaseObject},
    io::{IoBase, SysMem},
};
use bindings;
use core::{marker::*, ops::Deref, ptr};

/// The main interface for [`struct drm_framebuffer`].
///
/// # Invariants
///
/// - `self.0` is initialized for as long as this object is exposed to users.
/// - This type has an identical data layout to [`struct drm_framebuffer`]
///
/// [`struct drm_framebuffer`]: srctree/include/drm/drm_framebuffer.h
#[repr(transparent)]
pub struct Framebuffer<T: KmsDriver>(Opaque<bindings::drm_framebuffer>, PhantomData<T>);

// SAFETY:
// - `self.0` is initialized for as long as this object is exposed to users
// - `base` is initialized by DRM when `self.0` is initialized, thus `raw_mode_obj()` always returns
//   a valid pointer.
unsafe impl<T: KmsDriver> ModeObject for Framebuffer<T> {
    type Driver = T;

    fn drm_dev(&self) -> &Device<Self::Driver> {
        // SAFETY: `dev` points to an initialized `struct drm_device` for as long as this type is
        // initialized
        unsafe { Device::from_raw((*self.0.get()).dev) }
    }

    fn raw_mode_obj(&self) -> *mut bindings::drm_mode_object {
        // SAFETY: We don't expose Framebuffer<T> to users before its initialized, so `base` is
        // always initialized
        unsafe { &raw mut (*self.0.get()).base }
    }
}

// SAFETY: References to framebuffers are safe to be accessed from any thread
unsafe impl<T: KmsDriver> Send for Framebuffer<T> {}
// SAFETY: References to framebuffers are safe to be accessed from any thread
unsafe impl<T: KmsDriver> Sync for Framebuffer<T> {}

// For implementing ModeObject
impl<T: KmsDriver> Sealed for Framebuffer<T> {}

impl<T: KmsDriver> PartialEq for Framebuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0.get(), other.0.get())
    }
}
impl<T: KmsDriver> Eq for Framebuffer<T> {}

// SAFETY: DRM framebuffers use the refcount in their embedded mode object. The C get/put helpers
// operate on that refcount and release the object only after the last reference is dropped.
unsafe impl<T: KmsDriver> AlwaysRefCounted for Framebuffer<T> {
    fn inc_ref(&self) {
        // SAFETY: A shared reference proves the framebuffer and its refcount are live.
        unsafe { bindings::drm_framebuffer_get(self.0.get()) };
    }

    unsafe fn dec_ref(obj: core::ptr::NonNull<Self>) {
        // SAFETY: The caller transfers one live framebuffer reference to this method.
        unsafe { bindings::drm_framebuffer_put(obj.as_ref().0.get()) };
    }
}

/// A validated packed, linear framebuffer mapping backed by Lyude's shmem [`shmem::VMap`].
#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
pub struct FramebufferMapping<O, R>
where
    O: gem::DriverObject,
    R: Deref<Target = shmem::Object<O>>,
{
    map: shmem::VMap<O, R>,
    offset: usize,
    len: usize,
    pitch: usize,
    width: u32,
    height: u32,
    format: u32,
}

/// A framebuffer mapping borrowed from its backing object.
#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
pub type FramebufferVMap<'a, O> = FramebufferMapping<O, &'a shmem::Object<O>>;

/// A framebuffer mapping which owns a reference to its backing object.
///
/// This is suitable for a bounded scanout-registration cache: dropping it releases the mapping and
/// object reference, while retaining it keeps the validated CPU view stable across atomic commits.
#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
pub type FramebufferVMapOwned<O> = FramebufferMapping<O, ARef<shmem::Object<O>>>;

#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
struct PackedLayout {
    offset: usize,
    len: usize,
    pitch: usize,
}

#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
fn packed_layout(raw: &bindings::drm_framebuffer, object_size: usize) -> Result<PackedLayout> {
    if raw.format.is_null() {
        return Err(EINVAL);
    }

    // SAFETY: The caller supplies a live framebuffer, whose format descriptor remains valid.
    let format = unsafe { &*raw.format };
    if format.num_planes != 1 || raw.modifier != crate::drm::fourcc::FORMAT_MOD_LINEAR {
        return Err(EINVAL);
    }

    // Restrict this convenience adapter to ordinary packed scanlines. More complex block or tiled
    // layouts need a layout-specific API instead of pretending to be a byte raster.
    let block_width = unsafe { bindings::drm_format_info_block_width(raw.format, 0) };
    let block_height = unsafe { bindings::drm_format_info_block_height(raw.format, 0) };
    if block_width != 1 || block_height != 1 {
        return Err(EINVAL);
    }

    let min_pitch =
        usize::try_from(unsafe { bindings::drm_format_info_min_pitch(raw.format, 0, raw.width) })
            .map_err(|_| EOVERFLOW)?;
    let pitch = raw.pitches[0] as usize;
    if pitch < min_pitch {
        return Err(EINVAL);
    }

    let offset = raw.offsets[0] as usize;
    let len = pitch.checked_mul(raw.height as usize).ok_or(EOVERFLOW)?;
    let end = offset.checked_add(len).ok_or(EOVERFLOW)?;
    if end > object_size {
        return Err(EINVAL);
    }

    Ok(PackedLayout { offset, len, pitch })
}

#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
fn validate_object(
    raw: &bindings::drm_framebuffer,
    object: *mut bindings::drm_gem_object,
) -> Result {
    if object.is_null() {
        return Err(EINVAL);
    }
    // SAFETY: The object is non-null and live while its framebuffer owns it.
    let object = unsafe { &*object };
    if object.dev != raw.dev || !object.import_attach.is_null() {
        return Err(EINVAL);
    }
    Ok(())
}

#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
impl<O, R> FramebufferMapping<O, R>
where
    O: gem::DriverObject,
    R: Deref<Target = shmem::Object<O>>,
{
    /// Return the offset-adjusted pixel storage as a system-memory I/O view.
    pub fn view(&self) -> SysMem<'_, [u8]> {
        let base = (&self.map).as_view().as_ptr().cast::<u8>();
        // SAFETY: the mapping constructor checked `offset + len` against the object's size, and
        // borrowing `self` keeps the owning VMap alive for the returned view.
        let ptr = unsafe { core::ptr::slice_from_raw_parts_mut(base.add(self.offset), self.len) };
        // SAFETY: The range above is mapped, kernel-accessible system memory for this borrow.
        unsafe { SysMem::new(ptr) }
    }

    /// Return the validated line pitch in bytes.
    pub fn pitch(&self) -> usize {
        self.pitch
    }

    /// Return the visible width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Return the visible height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Return the DRM fourcc pixel format.
    pub fn format(&self) -> u32 {
        self.format
    }
}

impl<T: KmsDriver> Framebuffer<T> {
    /// Convert a raw pointer to a `struct drm_framebuffer` into a [`Framebuffer`]
    ///
    /// # Safety
    ///
    /// The caller guarantews that `ptr` points to a initialized `struct drm_framebuffer` for at
    /// least the entire lifetime of `'a`.
    #[inline]
    pub(super) unsafe fn from_raw<'a>(ptr: *const bindings::drm_framebuffer) -> &'a Self {
        // SAFETY: Our data layout is identical to drm_framebuffer
        unsafe { &*ptr.cast() }
    }

    /// Return an owned reference to this framebuffer.
    pub fn to_aref(&self) -> ARef<Self> {
        self.into()
    }

    /// Return the framebuffer width in pixels.
    pub fn width(&self) -> u32 {
        // SAFETY: The framebuffer is initialized via its type invariant.
        unsafe { (*self.0.get()).width }
    }

    /// Return the framebuffer height in pixels.
    pub fn height(&self) -> u32 {
        // SAFETY: The framebuffer is initialized via its type invariant.
        unsafe { (*self.0.get()).height }
    }

    /// Return the framebuffer's DRM fourcc pixel format.
    pub fn format(&self) -> u32 {
        // SAFETY: An initialized framebuffer has a valid format descriptor.
        unsafe { (*(*self.0.get()).format).format }
    }

    /// Return the pitch for `plane`, rejecting indices outside the format's actual plane count.
    pub fn pitch(&self, plane: usize) -> Result<u32> {
        // SAFETY: The framebuffer is initialized via its type invariant.
        let raw = unsafe { &*self.0.get() };
        if raw.format.is_null() {
            return Err(EINVAL);
        }
        // SAFETY: `format` is non-null and remains valid for the framebuffer's lifetime.
        if plane >= unsafe { (*raw.format).num_planes as usize } || plane >= raw.pitches.len() {
            return Err(EINVAL);
        }
        Ok(raw.pitches[plane])
    }

    /// Map a packed, single-plane, linear Rust shmem framebuffer.
    ///
    /// The returned view starts at the framebuffer plane's declared offset rather than the start
    /// of the GEM object. Multi-plane, imported, non-linear, block-compressed, undersized and
    /// cross-device objects are rejected.
    #[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
    pub fn vmap<O>(&self) -> Result<FramebufferVMap<'_, O>>
    where
        O: gem::DriverObject<Driver = T>,
        T: crate::drm::Driver<Object = shmem::Object<O>>,
    {
        // SAFETY: The framebuffer is initialized via its type invariant.
        let raw = unsafe { &*self.0.get() };
        let object_raw = raw.obj[0];
        validate_object(raw, object_raw)?;

        // SAFETY:
        // - `T::Object` is exactly `shmem::Object<O>` by the associated-type bound above.
        // - `validate_object` checked that this is a local, non-imported object owned by this
        //   framebuffer's instance of `T`.
        // - The framebuffer keeps its backing object alive for this borrow.
        let object = unsafe { <shmem::Object<O> as gem::IntoGEMObject>::from_raw(object_raw) };
        let layout = packed_layout(raw, object.size())?;

        Ok(FramebufferMapping {
            map: object.vmap()?,
            offset: layout.offset,
            len: layout.len,
            pitch: layout.pitch,
            width: raw.width,
            height: raw.height,
            // SAFETY: `packed_layout` rejected a null format pointer above.
            format: unsafe { (*raw.format).format },
        })
    }

    /// Returns the GEM object backing plane 0 of this framebuffer.
    ///
    /// A driver needs this to hand the buffer to a client, which is done by minting a handle for it
    /// in that client's file. The same type, ownership, import and device checks as [`Self::vmap`]
    /// apply, so the returned reference is known to belong to this driver.
    #[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
    pub fn object<O>(&self) -> Result<&shmem::Object<O>>
    where
        O: gem::DriverObject<Driver = T>,
        T: crate::drm::Driver<Object = shmem::Object<O>>,
    {
        // SAFETY: The framebuffer is initialized via its type invariant.
        let raw = unsafe { &*self.0.get() };
        let object_raw = raw.obj[0];
        validate_object(raw, object_raw)?;

        // SAFETY: `validate_object` established that `object_raw` is a live object of this
        // driver's type, and it is owned by the framebuffer for at least this borrow.
        Ok(unsafe { <shmem::Object<O> as gem::IntoGEMObject>::from_raw(object_raw) })
    }

    /// Map a packed, single-plane, linear Rust shmem framebuffer and retain its backing object.
    ///
    /// The validation is identical to [`Framebuffer::vmap`], but the returned mapping is not tied
    /// to this framebuffer borrow. It can therefore be retained in a bounded prepared-scanout
    /// cache and reused by later commits. The mapping itself keeps the GEM object alive.
    #[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
    pub fn owned_vmap<O>(&self) -> Result<FramebufferVMapOwned<O>>
    where
        O: gem::DriverObject<Driver = T>,
        T: crate::drm::Driver<Object = shmem::Object<O>>,
    {
        // SAFETY: The framebuffer is initialized via its type invariant.
        let raw = unsafe { &*self.0.get() };
        let object_raw = raw.obj[0];
        validate_object(raw, object_raw)?;

        // SAFETY: The same type, ownership, import, and device checks as `vmap` hold here. The
        // returned VMap takes its own object reference before this framebuffer borrow can end.
        let object = unsafe { <shmem::Object<O> as gem::IntoGEMObject>::from_raw(object_raw) };
        let layout = packed_layout(raw, object.size())?;

        Ok(FramebufferMapping {
            map: object.owned_vmap()?,
            offset: layout.offset,
            len: layout.len,
            pitch: layout.pitch,
            width: raw.width,
            height: raw.height,
            // SAFETY: `packed_layout` rejected a null format pointer above.
            format: unsafe { (*raw.format).format },
        })
    }
}

#[cfg(CONFIG_RUST_DRM_GEM_SHMEM_HELPER)]
#[kunit_tests(rust_drm_framebuffer)]
mod tests {
    use super::*;

    fn linear_fb(width: u32, height: u32, pitch: u32, offset: u32) -> bindings::drm_framebuffer {
        let mut fb = bindings::drm_framebuffer::default();
        // SAFETY: `XRGB8888` is a valid DRM fourcc and the returned descriptor has static lifetime.
        fb.format = unsafe { bindings::drm_format_info(crate::drm::fourcc::XRGB8888) };
        fb.modifier = crate::drm::fourcc::FORMAT_MOD_LINEAR;
        fb.width = width;
        fb.height = height;
        fb.pitches[0] = pitch;
        fb.offsets[0] = offset;
        fb
    }

    #[test]
    fn packed_layout_honours_nonzero_offset() -> Result {
        let fb = linear_fb(4, 2, 16, 128);
        let layout = packed_layout(&fb, 160)?;
        assert_eq!(layout.offset, 128);
        assert_eq!(layout.len, 32);
        Ok(())
    }

    #[test]
    fn packed_layout_rejects_too_small_object() {
        let fb = linear_fb(4, 2, 16, 128);
        assert!(packed_layout(&fb, 159).is_err());
    }

    #[test]
    fn packed_layout_rejects_multiple_planes() {
        let mut fb = linear_fb(4, 2, 16, 0);
        // SAFETY: `linear_fb` stored a non-null static format descriptor.
        let mut format = unsafe { *fb.format };
        format.num_planes = 2;
        fb.format = &raw const format;
        assert!(packed_layout(&fb, 32).is_err());
    }

    #[test]
    fn imported_object_is_rejected() {
        let mut fb = linear_fb(4, 2, 16, 0);
        let dev = ptr::NonNull::<bindings::drm_device>::dangling().as_ptr();
        fb.dev = dev;
        let mut object = bindings::drm_gem_object::default();
        object.dev = dev;
        object.import_attach = ptr::NonNull::<bindings::dma_buf_attachment>::dangling().as_ptr();
        assert!(validate_object(&fb, &raw mut object).is_err());
    }

    #[test]
    fn cross_device_object_is_rejected() {
        let mut first = core::mem::MaybeUninit::<bindings::drm_device>::uninit();
        let mut second = core::mem::MaybeUninit::<bindings::drm_device>::uninit();
        let mut fb = linear_fb(4, 2, 16, 0);
        fb.dev = first.as_mut_ptr();
        let mut object = bindings::drm_gem_object::default();
        object.dev = second.as_mut_ptr();
        assert!(validate_object(&fb, &raw mut object).is_err());
    }
}
