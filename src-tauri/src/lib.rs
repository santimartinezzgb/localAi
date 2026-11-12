use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, State};
use tokio::process::Command;
use tokio::sync::{oneshot, Mutex};
use walkdir::WalkDir;

const MAX_AGENT_STEPS: usize = 10;
const MAX_CONTENT_CHARS: usize = 15_000;
const TERMINAL_TIMEOUT_SECS: u64 = 30;

static CONFIRM_COUNTER: AtomicU64 = AtomicU64::new(1);

const KNOWN_TOOLS: &[&str] = &[
    "run_terminal",
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "grep_search",
    "web_search",
    "fetch_url",
];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Serialize, Clone, Debug)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    tools: Vec<Value>,
}

#[derive(Deserialize, Debug)]
struct OllamaChatResponse {
    message: Message,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,
}

#[derive(Serialize)]
pub struct AppResponse {
    pub reply: String,
    pub tps: f64,
    pub steps_executed: usize,
}

#[derive(Serialize, Clone)]
pub struct AgentStepEvent {
    pub step: usize,
    pub tag: String,
    pub tool: String,
    pub display: String,
    pub status: String,
    pub preview: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct ConfirmRequestEvent {
    pub id: String,
    pub tool: String,
    pub command_preview: String,
    pub details: String,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct LocalConfig {
    pub host: Option<String>,
    pub model: Option<String>,
    pub theme: Option<String>,
    pub safe_mode: Option<bool>,
}

pub struct AppState {
    pub client: reqwest::Client,
    pub pending_confirmations: Mutex<HashMap<String, oneshot::Sender<bool>>>,
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_chars).collect();
        format!(
            "{}\n\n... [TRUNCADO: Salida recortada a {} de {} caracteres para proteger la VRAM]",
            truncated, max_chars, char_count
        )
    }
}

fn build_system_prompt() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());

    format!(
        "Eres un agente autónomo de ingeniería de software y terminal (estilo Claude Code) ejecutándose localmente.\n\
        ENTORNO ACTUAL:\n\
        - Sistema Operativo: {} ({})\n\
        - Directorio de Trabajo Actual (CWD): {}\n\n\
        REGLAS ESTRICTAS DE OPERACIÓN:\n\
        1. Usa SIEMPRE las herramientas disponibles (`run_terminal`, `read_file`, `write_file`, `edit_file`, `list_directory`, `grep_search`, `web_search`, `fetch_url`) cuando necesites inspeccionar archivos, buscar código, ejecutar comandos o consultar información real.\n\
        2. NUNCA inventes rutas de archivos, contenidos de código ni salidas de terminal. Si no conoces la estructura del proyecto, usa primero `list_directory` o `grep_search`.\n\
        3. Antes de modificar un archivo con `edit_file` o `write_file`, léelo primero con `read_file` para conocer su contenido exacto.\n\
        4. Sé conciso, técnico y directo. Cuando termines de ejecutar las herramientas necesarias, responde al usuario con un resumen claro de los hallazgos o cambios realizados.",
        os, arch, cwd
    )
}

fn get_tools_schema() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "run_terminal",
                "description": "Ejecuta un comando en el shell del sistema (bash -c en Linux/macOS, cmd /C en Windows) con timeout de 30s y devuelve stdout, stderr y código de salida.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "El comando de terminal a ejecutar."
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Directorio de trabajo opcional donde ejecutar el comando (por defecto '.')."
                        }
                    },
                    "required": ["command"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Lee el contenido de un archivo local (truncado automáticamente a 15.000 caracteres).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Ruta relativa o absoluta del archivo a leer."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Crea o sobrescribe un archivo en el sistema de ficheros (creando directorios padres automáticamente si no existen).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Ruta del archivo a crear o sobrescribir."
                        },
                        "content": {
                            "type": "string",
                            "description": "Contenido completo que se escribirá en el archivo."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Reemplaza un bloque exacto de texto (target_text) por otro (replacement_text) dentro de un archivo existente.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Ruta del archivo a modificar."
                        },
                        "target_text": {
                            "type": "string",
                            "description": "Bloque exacto de texto existente que se desea reemplazar."
                        },
                        "replacement_text": {
                            "type": "string",
                            "description": "Nuevo texto que reemplazará a target_text."
                        }
                    },
                    "required": ["path", "target_text", "replacement_text"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "Lista archivos y carpetas de una ruta ignorando .git, node_modules y target.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Ruta del directorio a listar (por defecto '.')."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "grep_search",
                "description": "Busca texto o patrones regex dentro de los archivos de un directorio o archivo (ignorando .git, node_modules y target).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Texto o expresión regular a buscar."
                        },
                        "path": {
                            "type": "string",
                            "description": "Directorio o archivo donde realizar la búsqueda (por defecto '.')."
                        }
                    },
                    "required": ["pattern", "path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Realiza una búsqueda web en DuckDuckGo HTML y devuelve los primeros 5 resultados (título, URL y extracto).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Consulta de búsqueda web."
                        }
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "fetch_url",
                "description": "Descarga el contenido de una página web o documentación técnica y extrae su texto limpio sin HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL completa (http/https) a descargar."
                        }
                    },
                    "required": ["url"]
                }
            }
        }),
    ]
}

fn normalize_arguments(raw: &Value) -> Value {
    match raw {
        Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({})),
        Value::Object(_) => raw.clone(),
        _ => json!({}),
    }
}

fn parse_json_value_as_tool_calls(val: &Value) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    match val {
        Value::Object(map) => {
            if let Some(func_val) = map.get("function") {
                calls.extend(parse_json_value_as_tool_calls(func_val));
            } else if let Some(tc_arr) = map.get("tool_calls").and_then(|v| v.as_array()) {
                for item in tc_arr {
                    calls.extend(parse_json_value_as_tool_calls(item));
                }
            } else if let Some(name) = map.get("name").and_then(|v| v.as_str()) {
                if KNOWN_TOOLS.contains(&name) {
                    let args = map
                        .get("arguments")
                        .or_else(|| map.get("parameters"))
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    calls.push(ToolCall {
                        call_type: Some("function".to_string()),
                        function: ToolCallFunction {
                            name: name.to_string(),
                            arguments: normalize_arguments(&args),
                        },
                    });
                }
            }
        }
        Value::Array(arr) => {
            for item in arr {
                calls.extend(parse_json_value_as_tool_calls(item));
            }
        }
        _ => {}
    }
    calls
}

/// Fallback parser para modelos de 7B que emiten bloques JSON de herramientas en `content`
fn extract_fallback_tool_calls(content: &str) -> Vec<ToolCall> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // 1. Intentar parsear todo el content directamente si es JSON puro
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
            let parsed = parse_json_value_as_tool_calls(&val);
            if !parsed.is_empty() {
                return parsed;
            }
        }
    }

    // 2. Buscar etiquetas <tool_call>...</tool_call> (frecuente en Qwen2.5-Coder)
    if let Ok(re_tc) = Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>") {
        let mut results = Vec::new();
        for cap in re_tc.captures_iter(trimmed) {
            if let Some(m) = cap.get(1) {
                if let Ok(val) = serde_json::from_str::<Value>(m.as_str()) {
                    results.extend(parse_json_value_as_tool_calls(&val));
                }
            }
        }
        if !results.is_empty() {
            return results;
        }
    }

    // 3. Buscar bloques markdown ```json ... ``` o ``` ... ```
    if let Ok(re_fence) = Regex::new(r"(?s)```(?:json|tool_call)?\s*([\{\[].*?[\}\]])\s*```") {
        let mut results = Vec::new();
        for cap in re_fence.captures_iter(trimmed) {
            if let Some(m) = cap.get(1) {
                if let Ok(val) = serde_json::from_str::<Value>(m.as_str()) {
                    results.extend(parse_json_value_as_tool_calls(&val));
                }
            }
        }
        if !results.is_empty() {
            return results;
        }
    }

    // 4. Buscar objetos JSON balanceados en el texto que contengan "name" y "arguments"
    if trimmed.contains("\"name\"") && (trimmed.contains("\"arguments\"") || trimmed.contains("\"parameters\"")) {
        let mut results = Vec::new();
        let bytes = trimmed.as_bytes();
        let mut depth = 0i32;
        let mut start_idx: Option<usize> = None;
        let mut in_str = false;
        let mut escaped = false;

        for (i, &b) in bytes.iter().enumerate() {
            if in_str {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            if b == b'"' {
                in_str = true;
            } else if b == b'{' {
                if depth == 0 {
                    start_idx = Some(i);
                }
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = start_idx {
                        let candidate = &trimmed[start..=i];
                        if let Ok(val) = serde_json::from_str::<Value>(candidate) {
                            results.extend(parse_json_value_as_tool_calls(&val));
                        }
                    }
                    start_idx = None;
                }
            }
        }
        if !results.is_empty() {
            return results;
        }
    }

    Vec::new()
}

// =====================================================================
// IMPLEMENTACIÓN DE LAS 8 HERRAMIENTAS LOCALES (ESTILO CLAUDE CODE)
// =====================================================================

async fn tool_run_terminal(command: &str, cwd: Option<&str>) -> String {
    if command.trim().is_empty() {
        return "Error: El comando proporcionado está vacío.".to_string();
    }

    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("bash");
        c.args(["-c", command]);
        c
    };

    if let Some(dir) = cwd {
        let trimmed_dir = dir.trim();
        if !trimmed_dir.is_empty() {
            let p = Path::new(trimmed_dir);
            if p.exists() && p.is_dir() {
                cmd.current_dir(p);
            } else {
                return format!("Error: El directorio de trabajo '{}' no existe.", trimmed_dir);
            }
        }
    }

    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return format!("Error al lanzar el proceso de terminal: {}", e),
    };

    match tokio::time::timeout(
        Duration::from_secs(TERMINAL_TIMEOUT_SECS),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(output)) => {
            let code = output.status.code().unwrap_or(-1);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!(
                "Exit Code: {}\n--- STDOUT ---\n{}\n--- STDERR ---\n{}",
                code,
                stdout.trim_end(),
                stderr.trim_end()
            );
            truncate_chars(&combined, MAX_CONTENT_CHARS)
        }
        Ok(Err(e)) => format!("Error esperando la salida del comando: {}", e),
        Err(_) => format!(
            "Error: Timeout alcanzado ({}s). El comando '{}' tardó demasiado y fue terminado.",
            TERMINAL_TIMEOUT_SECS, command
        ),
    }
}

fn tool_read_file(path: &str) -> String {
    let p = Path::new(path.trim());
    if !p.exists() {
        return format!("Error: El archivo '{}' no existe.", path);
    }
    if p.is_dir() {
        return format!(
            "Error: '{}' es un directorio. Usa `list_directory` en su lugar.",
            path
        );
    }

    match fs::read_to_string(p) {
        Ok(content) => truncate_chars(&content, MAX_CONTENT_CHARS),
        Err(e) => format!("Error leyendo el archivo '{}': {}", path, e),
    }
}

fn tool_write_file(path: &str, content: &str) -> String {
    let p = Path::new(path.trim());
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                return format!(
                    "Error creando directorios padres para '{}': {}",
                    path, e
                );
            }
        }
    }

    match fs::write(p, content) {
        Ok(_) => {
            let lines = content.lines().count();
            let bytes = content.len();
            format!(
                "Archivo '{}' escrito correctamente ({} bytes, {} líneas).",
                path, bytes, lines
            )
        }
        Err(e) => format!("Error escribiendo el archivo '{}': {}", path, e),
    }
}

fn tool_edit_file(path: &str, target_text: &str, replacement_text: &str) -> String {
    let p = Path::new(path.trim());
    if !p.exists() {
        return format!("Error: El archivo '{}' no existe.", path);
    }
    if target_text.is_empty() {
        return "Error: `target_text` no puede estar vacío.".to_string();
    }

    let content = match fs::read_to_string(p) {
        Ok(c) => c,
        Err(e) => return format!("Error leyendo '{}': {}", path, e),
    };

    let occurrences = content.matches(target_text).count();
    if occurrences == 0 {
        return format!(
            "Error: No se encontró el bloque `target_text` exacto en '{}'. Asegúrate de leer el archivo con `read_file` primero.",
            path
        );
    }

    let updated = content.replace(target_text, replacement_text);
    match fs::write(p, &updated) {
        Ok(_) => format!(
            "Archivo '{}' editado correctamente ({} coincidencia(s) reemplazada(s)).",
            path, occurrences
        ),
        Err(e) => format!("Error guardando cambios en '{}': {}", path, e),
    }
}

fn is_ignored_dir(name: &str) -> bool {
    matches!(name, ".git" | "node_modules" | "target" | ".DS_Store")
}

fn tool_list_directory(path: &str) -> String {
    let target_path = if path.trim().is_empty() {
        "."
    } else {
        path.trim()
    };
    let root = Path::new(target_path);
    if !root.exists() {
        return format!("Error: La ruta '{}' no existe.", target_path);
    }
    if !root.is_dir() {
        return format!("Error: '{}' no es un directorio.", target_path);
    }

    let mut entries = Vec::new();
    let walker = WalkDir::new(root)
        .max_depth(3)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| {
            if e.depth() == 0 {
                return true;
            }
            let name = e.file_name().to_string_lossy();
            !is_ignored_dir(&name)
        });

    for entry in walker.flatten() {
        if entry.depth() == 0 {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .display()
            .to_string();
        if entry.file_type().is_dir() {
            entries.push(format!("[DIR]  {}/", rel));
        } else {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            entries.push(format!("[FILE] {} ({} B)", rel, size));
        }
        if entries.len() >= 400 {
            entries.push("... [Listado truncado a 400 elementos]".to_string());
            break;
        }
    }

    if entries.is_empty() {
        format!("El directorio '{}' está vacío.", target_path)
    } else {
        truncate_chars(&entries.join("\n"), MAX_CONTENT_CHARS)
    }
}

fn tool_grep_search(pattern: &str, path: &str) -> String {
    if pattern.trim().is_empty() {
        return "Error: El patrón de búsqueda no puede estar vacío.".to_string();
    }
    let target_path = if path.trim().is_empty() {
        "."
    } else {
        path.trim()
    };
    let root = Path::new(target_path);
    if !root.exists() {
        return format!("Error: La ruta '{}' no existe.", target_path);
    }

    let re = Regex::new(pattern).unwrap_or_else(|_| {
        let escaped = regex::escape(pattern);
        Regex::new(&escaped).expect("Regex escapado inválido")
    });

    let mut matches = Vec::new();
    let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
        if e.depth() == 0 {
            return true;
        }
        let name = e.file_name().to_string_lossy();
        !is_ignored_dir(&name)
    });

    'outer: for entry in walker.flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            if meta.len() > 2_000_000 {
                continue; // Ignorar archivos > 2 MB
            }
        }
        let content = match fs::read_to_string(entry.path()) {
            Ok(c) => c,
            Err(_) => continue, // Ignorar binarios o no UTF-8
        };

        let display_path = entry.path().display().to_string();
        for (idx, line) in content.lines().enumerate() {
            if re.is_match(line) {
                let snippet: String = line.trim().chars().take(220).collect();
                matches.push(format!("{}:{}: {}", display_path, idx + 1, snippet));
                if matches.len() >= 120 {
                    matches.push("... [Límite de 120 coincidencias alcanzado]".to_string());
                    break 'outer;
                }
            }
        }
    }

    if matches.is_empty() {
        format!(
            "No se encontraron coincidencias para '{}' en '{}'.",
            pattern, target_path
        )
    } else {
        truncate_chars(&matches.join("\n"), MAX_CONTENT_CHARS)
    }
}

fn clean_html_fragment(raw: &str) -> String {
    let without_tags = Regex::new(r"(?s)<[^>]+>")
        .map(|re| re.replace_all(raw, " ").to_string())
        .unwrap_or_else(|_| raw.to_string());

    let decoded = without_tags
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");

    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_ddg_target_url(raw_href: &str) -> String {
    let decoded_href = raw_href.replace("&amp;", "&");
    if let Some(pos) = decoded_href.find("uddg=") {
        let after = &decoded_href[pos + 5..];
        let end = after.find('&').unwrap_or(after.len());
        let encoded_url = &after[..end];
        if let Ok(decoded) = percent_encoding::percent_decode_str(encoded_url).decode_utf8() {
            return decoded.to_string();
        }
    }
    if decoded_href.starts_with("//") {
        format!("https:{}", decoded_href)
    } else {
        decoded_href
    }
}

async fn tool_web_search(client: &reqwest::Client, query: &str) -> String {
    if query.trim().is_empty() {
        return "Error: La consulta de búsqueda está vacía.".to_string();
    }

    let encoded_q: String =
        percent_encoding::utf8_percent_encode(query.trim(), percent_encoding::NON_ALPHANUMERIC)
            .to_string();
    let url = format!("https://html.duckduckgo.com/html/?q={}", encoded_q);

    let res = match client
        .get(&url)
        .header(
            "User-Agent",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
        .timeout(Duration::from_secs(15))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("Error de red al buscar en DuckDuckGo: {}", e),
    };

    let html = match res.text().await {
        Ok(t) => t,
        Err(e) => return format!("Error leyendo respuesta de DuckDuckGo: {}", e),
    };

    let re_title = Regex::new(
        r#"(?s)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#,
    )
    .expect("Regex válido");
    let re_snippet = Regex::new(
        r#"(?s)<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>"#,
    )
    .expect("Regex válido");

    let snippets: Vec<String> = re_snippet
        .captures_iter(&html)
        .filter_map(|c| c.get(1).map(|m| clean_html_fragment(m.as_str())))
        .collect();

    let mut results = Vec::new();
    for (idx, cap) in re_title.captures_iter(&html).take(5).enumerate() {
        let raw_href = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let raw_title = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        let clean_url = extract_ddg_target_url(raw_href);
        let title = clean_html_fragment(raw_title);
        let snippet = snippets
            .get(idx)
            .cloned()
            .unwrap_or_else(|| "Sin descripción disponible.".to_string());

        results.push(format!(
            "{}. {}\n   URL: {}\n   Extracto: {}",
            idx + 1,
            title,
            clean_url,
            snippet
        ));
    }

    if results.is_empty() {
        format!("No se encontraron resultados en DuckDuckGo para '{}'.", query)
    } else {
        results.join("\n\n")
    }
}

async fn tool_fetch_url(client: &reqwest::Client, url: &str) -> String {
    let target = url.trim();
    if !target.starts_with("http://") && !target.starts_with("https://") {
        return format!("Error: La URL '{}' debe comenzar con http:// o https://", target);
    }

    let res = match client
        .get(target)
        .header(
            "User-Agent",
            "Mozilla/5.0 (compatible; localAiAgent/2.0; +https://github.com)",
        )
        .timeout(Duration::from_secs(20))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("Error descargando '{}': {}", target, e),
    };

    if !res.status().is_success() {
        return format!("Error HTTP {} al descargar '{}'", res.status(), target);
    }

    let body = match res.text().await {
        Ok(t) => t,
        Err(e) => return format!("Error leyendo el cuerpo de '{}': {}", target, e),
    };

    // Eliminar bloques script, style, noscript y svg
    let cleaned = Regex::new(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>")
        .map(|re| re.replace_all(&body, " ").to_string())
        .unwrap_or(body);

    let text = clean_html_fragment(&cleaned);
    if text.is_empty() {
        "La página fue descargada pero no contiene texto extraíble.".to_string()
    } else {
        truncate_chars(&text, MAX_CONTENT_CHARS)
    }
}

fn format_tool_badge(tool_name: &str, args: &Value) -> (String, String, String) {
    match tool_name {
        "run_terminal" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let cwd = args.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");
            (
                "[EXEC]".to_string(),
                format!("run_terminal: {} (cwd: {})", cmd, cwd),
                format!("Comando: {}\nDirectorio: {}", cmd, cwd),
            )
        }
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            (
                "[READ]".to_string(),
                format!("read_file: {}", path),
                format!("Archivo: {}", path),
            )
        }
        "write_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let preview: String = content.chars().take(300).collect();
            (
                "[WRITE]".to_string(),
                format!("write_file: {} ({} bytes)", path, content.len()),
                format!("Archivo: {}\nVista previa:\n{}", path, preview),
            )
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let target = args.get("target_text").and_then(|v| v.as_str()).unwrap_or("");
            let replacement = args
                .get("replacement_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let target_prev: String = target.chars().take(140).collect();
            let repl_prev: String = replacement.chars().take(140).collect();
            (
                "[EDIT]".to_string(),
                format!("edit_file: {}", path),
                format!(
                    "Archivo: {}\nReemplazar:\n{}\nPor:\n{}",
                    path, target_prev, repl_prev
                ),
            )
        }
        "list_directory" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            (
                "[LIST]".to_string(),
                format!("list_directory: {}", path),
                format!("Directorio: {}", path),
            )
        }
        "grep_search" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            (
                "[GREP]".to_string(),
                format!("grep_search: '{}' en {}", pattern, path),
                format!("Patrón: {}\nRuta: {}", pattern, path),
            )
        }
        "web_search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            (
                "[SEARCH]".to_string(),
                format!("web_search: {}", query),
                format!("Búsqueda DuckDuckGo: {}", query),
            )
        }
        "fetch_url" => {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
            (
                "[FETCH]".to_string(),
                format!("fetch_url: {}", url),
                format!("URL: {}", url),
            )
        }
        other => (
            "[TOOL]".to_string(),
            format!("{}: {}", other, args),
            args.to_string(),
        ),
    }
}

fn requires_safe_mode_confirmation(tool_name: &str) -> bool {
    matches!(tool_name, "run_terminal" | "write_file" | "edit_file")
}

async fn request_user_confirmation(
    app: &AppHandle,
    state: &AppState,
    step: usize,
    tool_name: &str,
    display: &str,
    details: &str,
) -> bool {
    let confirm_id = format!(
        "confirm-{}-{}",
        step,
        CONFIRM_COUNTER.fetch_add(1, Ordering::SeqCst)
    );
    let (tx, rx) = oneshot::channel::<bool>();

    {
        let mut pending = state.pending_confirmations.lock().await;
        pending.insert(confirm_id.clone(), tx);
    }

    let _ = app.emit(
        "agent-confirm-request",
        ConfirmRequestEvent {
            id: confirm_id.clone(),
            tool: tool_name.to_string(),
            command_preview: display.to_string(),
            details: details.to_string(),
        },
    );

    match tokio::time::timeout(Duration::from_secs(300), rx).await {
        Ok(Ok(approved)) => approved,
        _ => {
            let mut pending = state.pending_confirmations.lock().await;
            pending.remove(&confirm_id);
            false
        }
    }
}

async fn execute_tool_call(
    client: &reqwest::Client,
    tool_name: &str,
    args: &Value,
) -> String {
    match tool_name {
        "run_terminal" => {
            let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let cwd = args.get("cwd").and_then(|v| v.as_str());
            tool_run_terminal(command, cwd).await
        }
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            tool_read_file(path)
        }
        "write_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            tool_write_file(path, content)
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let target_text = args
                .get("target_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let replacement_text = args
                .get("replacement_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            tool_edit_file(path, target_text, replacement_text)
        }
        "list_directory" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            tool_list_directory(path)
        }
        "grep_search" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            tool_grep_search(pattern, path)
        }
        "web_search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            tool_web_search(client, query).await
        }
        "fetch_url" => {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
            tool_fetch_url(client, url).await
        }
        unknown => format!("Error: Herramienta desconocida '{}'.", unknown),
    }
}

#[tauri::command]
async fn confirm_tool_execution(
    state: State<'_, AppState>,
    id: String,
    approved: bool,
) -> Result<(), String> {
    let mut pending = state.pending_confirmations.lock().await;
    if let Some(tx) = pending.remove(&id) {
        let _ = tx.send(approved);
        Ok(())
    } else {
        Err(format!("Solicitud de confirmación '{}' no encontrada o expirada.", id))
    }
}

fn config_file_path() -> PathBuf {
    PathBuf::from("config.json")
}

pub fn normalize_host_url(host: &str) -> Result<String, String> {
    let mut h = host.trim().trim_end_matches('/').to_string();
    if h.is_empty() {
        return Err("La dirección del Host está vacía. Introduce la URL de tu nodo Ollama (ej. http://100.x.x.x:11434 o http://localhost:11434)".to_string());
    }
    if !h.starts_with("http://") && !h.starts_with("https://") {
        h = format!("http://{}", h);
    }

    if let Ok(mut parsed) = url::Url::parse(&h) {
        if parsed.port().is_none() {
            let _ = parsed.set_port(Some(11434));
            h = parsed.to_string().trim_end_matches('/').to_string();
        }
    } else {
        return Err(format!(
            "URL de Host inválida '{}'. Formato esperado: http://<IP-o-Host>:11434",
            host
        ));
    }

    Ok(h)
}

#[tauri::command]
fn load_local_config() -> LocalConfig {
    let path = config_file_path();
    if let Ok(content) = fs::read_to_string(path) {
        if let Ok(mut cfg) = serde_json::from_str::<LocalConfig>(&content) {
            if let Some(ref h) = cfg.host {
                if let Ok(norm) = normalize_host_url(h) {
                    cfg.host = Some(norm);
                }
            }
            return cfg;
        }
    }
    LocalConfig::default()
}

#[tauri::command]
fn save_local_config(mut config: LocalConfig) -> Result<(), String> {
    if let Some(ref h) = config.host {
        if let Ok(norm) = normalize_host_url(h) {
            config.host = Some(norm);
        }
    }
    let path = config_file_path();
    let json_str = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Error serializando config.json: {}", e))?;
    fs::write(path, json_str).map_err(|e| format!("Error guardando config.json: {}", e))
}

#[tauri::command]
async fn chat_ollama(
    app: AppHandle,
    state: State<'_, AppState>,
    host: String,
    model: String,
    messages: Vec<Message>,
    #[allow(unused)] safe_mode: Option<bool>,
) -> Result<AppResponse, String> {
    let is_safe_mode = safe_mode.unwrap_or(true);
    let host_norm = normalize_host_url(&host)?;
    let url = format!("{}/api/chat", host_norm);
    let tools = get_tools_schema();

    // Construir historial e inyectar el System Prompt al inicio si no está presente
    let mut working_messages = Vec::with_capacity(messages.len() + 4);
    let has_system = messages.first().map(|m| m.role == "system").unwrap_or(false);
    if !has_system {
        working_messages.push(Message {
            role: "system".to_string(),
            content: build_system_prompt(),
            tool_calls: None,
        });
    }
    working_messages.extend(messages);

    let mut total_eval_count: u64 = 0;
    let mut total_eval_duration: u64 = 0;
    let mut steps_executed: usize = 0;
    let overall_start = Instant::now();

    for step in 1..=MAX_AGENT_STEPS {
        let payload = OllamaChatRequest {
            model: model.clone(),
            messages: working_messages.clone(),
            stream: false,
            tools: tools.clone(),
        };

        let res = state
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    format!("No se pudo conectar con Ollama en '{}'. Verifica que el nodo remoto esté encendido y Tailscale activo ({}).", url, e)
                } else if e.is_timeout() {
                    format!("Tiempo de espera agotado al conectar con Ollama en '{}' ({}).", url, e)
                } else {
                    format!("Error de red con Ollama en '{}': {}", url, e)
                }
            })?;

        if !res.status().is_success() {
            let status = res.status();
            let err_body = res.text().await.unwrap_or_default();
            return Err(format!(
                "Error del servidor Ollama (HTTP {}): {}",
                status, err_body
            ));
        }

        let data: OllamaChatResponse = res
            .json()
            .await
            .map_err(|e| format!("Error procesando respuesta JSON de Ollama: {}", e))?;

        total_eval_count += data.eval_count;
        total_eval_duration += data.eval_duration;

        // Detectar llamadas a herramientas nativas o mediante fallback JSON en content
        let mut detected_calls: Vec<ToolCall> = Vec::new();
        let mut used_fallback = false;

        if let Some(ref tc) = data.message.tool_calls {
            if !tc.is_empty() {
                for call in tc {
                    detected_calls.push(ToolCall {
                        call_type: call.call_type.clone().or_else(|| Some("function".to_string())),
                        function: ToolCallFunction {
                            name: call.function.name.clone(),
                            arguments: normalize_arguments(&call.function.arguments),
                        },
                    });
                }
            }
        }

        if detected_calls.is_empty() {
            let fallback_calls = extract_fallback_tool_calls(&data.message.content);
            if !fallback_calls.is_empty() {
                detected_calls = fallback_calls;
                used_fallback = true;
            }
        }

        // Si no hay llamadas a herramientas, hemos llegado a la respuesta final
        if detected_calls.is_empty() {
            let elapsed = overall_start.elapsed().as_secs_f64();
            let tps = if total_eval_duration > 0 {
                (total_eval_count as f64) / (total_eval_duration as f64 / 1_000_000_000.0)
            } else if elapsed > 0.0 {
                (total_eval_count as f64) / elapsed
            } else {
                0.0
            };

            return Ok(AppResponse {
                reply: data.message.content,
                tps,
                steps_executed,
            });
        }

        // Registrar el turno del asistente antes de anexar respuestas de herramientas
        working_messages.push(Message {
            role: "assistant".to_string(),
            content: data.message.content.clone(),
            tool_calls: if used_fallback {
                None
            } else {
                Some(detected_calls.clone())
            },
        });

        for call in detected_calls {
            steps_executed += 1;
            let tool_name = call.function.name.as_str();
            let args = &call.function.arguments;
            let (tag, display, details) = format_tool_badge(tool_name, args);

            // Si el Modo Seguro está activo y la herramienta es crítica, solicitar confirmación
            if is_safe_mode && requires_safe_mode_confirmation(tool_name) {
                let _ = app.emit(
                    "agent-step",
                    AgentStepEvent {
                        step: steps_executed,
                        tag: tag.clone(),
                        tool: tool_name.to_string(),
                        display: format!("{} (Esperando confirmación...)", display),
                        status: "waiting_confirm".to_string(),
                        preview: None,
                    },
                );

                let approved = request_user_confirmation(
                    &app,
                    &state,
                    steps_executed,
                    tool_name,
                    &display,
                    &details,
                )
                .await;

                if !approved {
                    let denied_msg = format!(
                        "Ejecución de `{}` cancelada/denegada por el usuario en Modo Seguro.",
                        tool_name
                    );
                    let _ = app.emit(
                        "agent-step",
                        AgentStepEvent {
                            step: steps_executed,
                            tag: "[DENIED]".to_string(),
                            tool: tool_name.to_string(),
                            display: format!("{} -> Cancelado por el usuario", display),
                            status: "denied".to_string(),
                            preview: Some(denied_msg.clone()),
                        },
                    );

                    working_messages.push(Message {
                        role: "tool".to_string(),
                        content: denied_msg,
                        tool_calls: None,
                    });
                    continue;
                }
            }

            // Emitir evento de inicio de ejecución en el monitor CRT
            let _ = app.emit(
                "agent-step",
                AgentStepEvent {
                    step: steps_executed,
                    tag: tag.clone(),
                    tool: tool_name.to_string(),
                    display: display.clone(),
                    status: "running".to_string(),
                    preview: None,
                },
            );

            let tool_output = execute_tool_call(&state.client, tool_name, args).await;
            let short_preview: String = tool_output
                .lines()
                .take(2)
                .collect::<Vec<_>>()
                .join(" | ")
                .chars()
                .take(140)
                .collect();

            let _ = app.emit(
                "agent-step",
                AgentStepEvent {
                    step: steps_executed,
                    tag,
                    tool: tool_name.to_string(),
                    display,
                    status: "completed".to_string(),
                    preview: Some(short_preview),
                },
            );

            // Añadir el resultado al historial con role="tool" (y si fue fallback, contexto claro)
            let formatted_tool_content = if used_fallback {
                format!(
                    "[Resultado de herramienta `{}`]:\n{}",
                    tool_name, tool_output
                )
            } else {
                tool_output
            };

            working_messages.push(Message {
                role: "tool".to_string(),
                content: formatted_tool_content,
                tool_calls: None,
            });
        }

        if step == MAX_AGENT_STEPS {
            let _ = app.emit(
                "agent-step",
                AgentStepEvent {
                    step: steps_executed,
                    tag: "[WARN]".to_string(),
                    tool: "agent_loop".to_string(),
                    display: format!(
                        "Límite de seguridad alcanzado ({} pasos máximos por turno).",
                        MAX_AGENT_STEPS
                    ),
                    status: "completed".to_string(),
                    preview: None,
                },
            );
        }
    }

    // Si alcanzó los 10 pasos sin respuesta de texto plano, hacer una última llamada sin herramientas para resumir
    working_messages.push(Message {
        role: "user".to_string(),
        content: "Has alcanzado el límite máximo de 10 pasos de herramientas en este turno. Por favor, proporciona ahora la respuesta final o resumen con la información obtenida hasta el momento.".to_string(),
        tool_calls: None,
    });

    let final_payload = OllamaChatRequest {
        model,
        messages: working_messages,
        stream: false,
        tools: Vec::new(),
    };

    let res = state
        .client
        .post(&url)
        .json(&final_payload)
        .send()
        .await
        .map_err(|e| format!("Error de red en resumen final con Ollama en '{}': {}", url, e))?;

    let data: OllamaChatResponse = res
        .json()
        .await
        .map_err(|e| format!("Error procesando JSON final: {}", e))?;

    total_eval_count += data.eval_count;
    total_eval_duration += data.eval_duration;
    let elapsed = overall_start.elapsed().as_secs_f64();
    let tps = if total_eval_duration > 0 {
        (total_eval_count as f64) / (total_eval_duration as f64 / 1_000_000_000.0)
    } else if elapsed > 0.0 {
        (total_eval_count as f64) / elapsed
    } else {
        0.0
    };

    Ok(AppResponse {
        reply: data.message.content,
        tps,
        steps_executed,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let client = reqwest::Client::builder()
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .connect_timeout(Duration::from_secs(12))
        .build()
        .expect("Error al inicializar el cliente HTTP");

    tauri::Builder::default()
        .manage(AppState {
            client,
            pending_confirmations: Mutex::new(HashMap::new()),
        })
        .invoke_handler(tauri::generate_handler![
            chat_ollama,
            confirm_tool_execution,
            load_local_config,
            save_local_config
        ])
        .run(tauri::generate_context!())
        .expect("Error al iniciar la aplicación Tauri");
}
