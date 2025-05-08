## 静态链接

主要是处理 [ort](https://ort.pyke.io/) 的静态链接，[参考文档](https://ort.pyke.io/setup/linking#static-linking)。

###  Windows 示例：

在 [onnxruntime-build](https://github.com/supertone-inc/onnxruntime-build/releases) 下载 Windows 平台的 lib。

解压后将 lib 文件夹放到项目根目录下（注意是解压包里的 lib 文件夹），然后配置 .cargo/config.toml：

```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-Ctarget-feature=+crt-static"]
[target.i686-pc-windows-msvc]
rustflags = ["-Ctarget-feature=+crt-static"]

[env]
ORT_LIB_LOCATION = "./lib"
```

环境变量 `ORT_LIB_LOCATION` 可以自由配置，路径应指向文件夹。
