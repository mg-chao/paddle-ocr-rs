use ort::execution_providers::{
    CANNExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider,
    ExecutionProvider, ExecutionProviderDispatch,
};

use crate::{
    config::ProviderPreference,
    error::{PaddleOcrError, Result},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedExecutionProvider {
    Cpu,
    Cuda,
    DirectMl,
    Cann,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderResolution {
    pub requested: ProviderPreference,
    pub resolved: ResolvedExecutionProvider,
    pub fallback_used: bool,
}

#[derive(Debug)]
pub struct ProviderChain {
    pub providers: Vec<ExecutionProviderDispatch>,
    pub resolution: ProviderResolution,
}

pub fn resolve_execution_providers(
    preference: &ProviderPreference,
    enable_cpu_mem_arena: bool,
    fail_if_provider_unavailable: bool,
) -> Result<ProviderChain> {
    let cpu_provider = || {
        CPUExecutionProvider::default()
            .with_arena_allocator(enable_cpu_mem_arena)
            .build()
    };

    match preference {
        ProviderPreference::Cpu => Ok(ProviderChain {
            providers: vec![cpu_provider()],
            resolution: ProviderResolution {
                requested: ProviderPreference::Cpu,
                resolved: ResolvedExecutionProvider::Cpu,
                fallback_used: false,
            },
        }),
        ProviderPreference::Cuda { device_id } => resolve_cuda_execution_providers(
            *device_id,
            cpu_provider(),
            fail_if_provider_unavailable,
        ),
        ProviderPreference::DirectMl { device_id } => resolve_directml_execution_providers(
            *device_id,
            cpu_provider(),
            fail_if_provider_unavailable,
        ),
        ProviderPreference::Cann { device_id } => resolve_cann_execution_providers(
            *device_id,
            cpu_provider(),
            fail_if_provider_unavailable,
        ),
    }
}

fn device_id_to_i32(provider_name: &str, device_id: usize) -> Result<i32> {
    i32::try_from(device_id).map_err(|_| {
        PaddleOcrError::Config(format!(
            "invalid {provider_name} device_id {device_id}: value exceeds i32 range"
        ))
    })
}

fn resolve_cuda_execution_providers(
    device_id: usize,
    cpu_provider: ExecutionProviderDispatch,
    fail_if_provider_unavailable: bool,
) -> Result<ProviderChain> {
    let device_id_i32 = device_id_to_i32("CUDA", device_id)?;
    let cuda_provider = CUDAExecutionProvider::default().with_device_id(device_id_i32);
    let is_available = cuda_provider.is_available()?;
    let resolution = decide_provider_resolution(
        ProviderPreference::Cuda { device_id },
        ResolvedExecutionProvider::Cuda,
        is_available,
        fail_if_provider_unavailable,
    )?;

    if resolution.fallback_used {
        Ok(ProviderChain {
            providers: vec![cpu_provider],
            resolution,
        })
    } else {
        Ok(ProviderChain {
            providers: vec![cuda_provider.build(), cpu_provider],
            resolution,
        })
    }
}

fn resolve_directml_execution_providers(
    device_id: usize,
    cpu_provider: ExecutionProviderDispatch,
    fail_if_provider_unavailable: bool,
) -> Result<ProviderChain> {
    let device_id_i32 = device_id_to_i32("DirectML", device_id)?;

    let dml_provider = DirectMLExecutionProvider::default().with_device_id(device_id_i32);
    let is_available = dml_provider.is_available()?;
    let resolution = decide_provider_resolution(
        ProviderPreference::DirectMl { device_id },
        ResolvedExecutionProvider::DirectMl,
        is_available,
        fail_if_provider_unavailable,
    )?;

    if resolution.fallback_used {
        Ok(ProviderChain {
            providers: vec![cpu_provider],
            resolution,
        })
    } else {
        Ok(ProviderChain {
            providers: vec![dml_provider.build(), cpu_provider],
            resolution,
        })
    }
}

fn resolve_cann_execution_providers(
    device_id: usize,
    cpu_provider: ExecutionProviderDispatch,
    fail_if_provider_unavailable: bool,
) -> Result<ProviderChain> {
    let device_id_i32 = device_id_to_i32("CANN", device_id)?;
    let cann_provider = CANNExecutionProvider::default().with_device_id(device_id_i32);
    let is_available = cann_provider.is_available()?;
    let resolution = decide_provider_resolution(
        ProviderPreference::Cann { device_id },
        ResolvedExecutionProvider::Cann,
        is_available,
        fail_if_provider_unavailable,
    )?;

    if resolution.fallback_used {
        Ok(ProviderChain {
            providers: vec![cpu_provider],
            resolution,
        })
    } else {
        Ok(ProviderChain {
            providers: vec![cann_provider.build(), cpu_provider],
            resolution,
        })
    }
}

fn decide_provider_resolution(
    requested: ProviderPreference,
    preferred: ResolvedExecutionProvider,
    preferred_is_available: bool,
    fail_if_provider_unavailable: bool,
) -> Result<ProviderResolution> {
    if preferred_is_available {
        return Ok(ProviderResolution {
            requested,
            resolved: preferred,
            fallback_used: false,
        });
    }

    if fail_if_provider_unavailable {
        return Err(PaddleOcrError::Config(format!(
            "requested execution provider {} is unavailable and fail_if_provider_unavailable=true",
            format_provider_preference(requested)
        )));
    }

    Ok(ProviderResolution {
        requested,
        resolved: ResolvedExecutionProvider::Cpu,
        fallback_used: true,
    })
}

fn format_provider_preference(preference: ProviderPreference) -> String {
    match preference {
        ProviderPreference::Cpu => "cpu".to_string(),
        ProviderPreference::Cuda { device_id } => format!("cuda(device_id={device_id})"),
        ProviderPreference::DirectMl { device_id } => {
            format!("directml(device_id={device_id})")
        }
        ProviderPreference::Cann { device_id } => format!("cann(device_id={device_id})"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ResolvedExecutionProvider, decide_provider_resolution, resolve_execution_providers,
    };
    use crate::config::ProviderPreference;

    #[test]
    fn directml_preference_has_cpu_fallback() {
        let providers = resolve_execution_providers(
            &ProviderPreference::DirectMl { device_id: 0 },
            false,
            false,
        )
        .expect("provider resolution should not fail");
        assert!(
            !providers.providers.is_empty(),
            "provider chain must contain at least CPU fallback"
        );
    }

    #[test]
    fn cuda_preference_has_cpu_fallback() {
        let providers =
            resolve_execution_providers(&ProviderPreference::Cuda { device_id: 0 }, false, false)
                .expect("provider resolution should not fail");
        assert!(
            !providers.providers.is_empty(),
            "provider chain must contain at least CPU fallback"
        );
    }

    #[test]
    fn cann_preference_has_cpu_fallback() {
        let providers =
            resolve_execution_providers(&ProviderPreference::Cann { device_id: 0 }, false, false)
                .expect("provider resolution should not fail");
        assert!(
            !providers.providers.is_empty(),
            "provider chain must contain at least CPU fallback"
        );
    }

    #[test]
    fn strict_mode_errors_when_provider_is_unavailable() {
        let err = decide_provider_resolution(
            ProviderPreference::Cuda { device_id: 0 },
            ResolvedExecutionProvider::Cuda,
            false,
            true,
        )
        .expect_err("strict mode should reject unavailable provider");
        assert!(
            err.to_string()
                .contains("fail_if_provider_unavailable=true")
        );
    }

    #[test]
    fn non_strict_mode_falls_back_to_cpu_when_provider_is_unavailable() {
        let resolution = decide_provider_resolution(
            ProviderPreference::DirectMl { device_id: 2 },
            ResolvedExecutionProvider::DirectMl,
            false,
            false,
        )
        .expect("fallback should succeed");
        assert!(resolution.fallback_used);
        assert_eq!(resolution.resolved, ResolvedExecutionProvider::Cpu);
    }
}
