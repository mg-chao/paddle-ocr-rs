use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OnnxRuntimeCompat {
    pub(crate) intra_op_num_threads: Option<usize>,
    pub(crate) inter_op_num_threads: Option<usize>,
    pub(crate) auto_tune_threads: Option<bool>,
    pub(crate) rayon_threads: Option<usize>,
    pub(crate) enable_cpu_mem_arena: Option<bool>,
    pub(crate) fail_if_provider_unavailable: Option<bool>,
    pub(crate) use_cuda: Option<bool>,
    pub(crate) use_dml: Option<bool>,
    pub(crate) use_cann: Option<bool>,
    pub(crate) cuda_ep_cfg: Option<DeviceCfg>,
    pub(crate) dm_ep_cfg: Option<DeviceCfg>,
    pub(crate) cann_ep_cfg: Option<DeviceCfg>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DeviceCfg {
    pub(crate) device_id: Option<usize>,
}
