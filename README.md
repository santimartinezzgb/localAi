# localAi

Un agente autónomo de programación y terminal para tu ordenador alimentado por modelos locales o remotos con **Ollama**.

---

## Inicio Rápido (3 pasos)

### 1. Requisitos Previos
- **[Ollama](https://ollama.com/)** instalado (en tu PC o en un servidor remoto/Tailscale).
- **[Rust](https://rustup.rs/)** instalado.
- En Linux (Ubuntu/Debian), instala las librerías necesarias:
  ```bash
  sudo apt update && sudo apt install -y build-essential curl wget libwebkit2gtk-4.1-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
  ```

### 2. Descargar el Modelo
En la máquina donde tengas Ollama, descarga el modelo recomendado:
```bash
ollama run qwen2.5-coder:7b
```

### 3. Ejecutar la Aplicación
Entra en la carpeta del proyecto y lánzala:
```bash
cd src-tauri
cargo run
```

---

## Conexión con Ollama

En la barra superior de la aplicación encontrarás el campo **HOST**:

- **Si Ollama está en el mismo ordenador**: Déjalo en `http://localhost:11434`.
- **Si Ollama está en otra máquina o Tailscale**: Introduce la IP o nombre de red (ej. `http://100.x.x.x:11434` o simplemente la IP `100.x.x.x`). La aplicación autocompleta el puerto y guarda la dirección automáticamente en local para que no tengas que escribirla de nuevo.

---

## ¿Qué puede hacer el Agente?

Puedes pedirle cosas en lenguaje natural y él usará sus herramientas locales para resolverlas:

1. **Ejecutar comandos**: *"Compila el proyecto y muéstrame los errores"*.
2. **Leer y buscar código**: *"Busca dónde está definida la función X"* o *"Revisa el archivo index.html"*.
3. **Modificar y crear archivos**: *"Corrige el bug en la línea 40 de este archivo"*.
4. **Búsqueda web**: *"Busca en internet la documentación de Tauri v2 sobre eventos"*.

> **MODO SEGURO**: Viene activado por defecto con un interruptor en la cabecera. Cada vez que el agente intente ejecutar un comando en la consola o modificar un archivo, te pedirá confirmación antes de hacerlo.

---

## Temas Visuales

Pulsa el botón **Ajustes** en la esquina superior para alternar entre 4 estilos:
- **OpenCode**: Oscuro y minimalista (estilo Cursor / VS Code).
- **Tipo Claude**: Tonos cálidos terracota.
- **Terminal Hacker**: Monitor CRT fósforo verde con scanlines.
- **Retro 90**: Clásico Windows 95 con bordes biselados.

---

## Generar el Ejecutable Final

Para crear el binario optimizado para tu sistema:
```bash
cd src-tauri
cargo build --release
```
El ejecutable resultante estará en `src-tauri/target/release/localai`.
