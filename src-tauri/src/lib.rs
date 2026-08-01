// Tauri 应用入口：构建窗口并加载前端资源
// 前端通过 HTTP 调用独立运行的 FastAPI 后端（端口由 config/server_config.json 配置），此处仅作壳层

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用时出错");
}
