light_palette = {
    # --- 三态底色 ---
    "$primary":           "#111111",  # 默认黑
    "$primary-hover":     "#f5f6fa",  # Hover 背景（亮灰）
    "$primary-pressed":   "#e7e9ee",  # Pressed 背景（更亮）

    # --- 描边 & 文字 ---
    "$btn-text":            "#ffffff",  # 默认白字
    "$btn-hover-border":    "#3b82f6",  # Hover 边：科技蓝
    "$btn-hover-text":      "#111111",  # Hover 字：黑
    "$btn-pressed-border":  "#2563eb",  # Pressed 边：深蓝
    "$btn-pressed-text":    "#111111",  # Pressed 字：黑

    "$border":             "#e5e7eb",
    "$bg":                 "#ffffff",
    "$surface":            "#ffffff",

    "$sidebar-bg":         "#f5f6f8",
    "$sidebar-hover":      "rgba(0,0,0,0.05)",

    "$text":               "#111111",
    "$text-light":         "#6b7280",

    "$disabled-bg":        "#d1d5db",
    "$disabled-border":    "#d1d5db",
    "$disabled-text":      "#f9fafb",

    "$input-bg":           "#f1f3f5",
    "$table-header-bg":    "#fafafa",

    "$scrollbar":          "rgba(0,0,0,0.25)",
    "$scrollbar-hover":    "rgba(1,0,0,0.35)",
    "$scrollbar-pressed":  "rgba(1,0,0,0.45)",

    "$tooltip-bg":         "#111111",
    "$tooltip-text":       "#ffffff",

    "$card-shadow":        "0 1px 4px rgba(0, 0, 0, 0.05)",
    "$radius":             "12px",
}

def get_sheet(style_color:str):
    # 读取 QSS 模板
    with open("style_sheet_template.qss", encoding="utf-8") as f:
        template = f.read()

    palette = {}
    if style_color == "light":
        palette = light_palette

    # 按变量名长度逆序替换，避免 $primary 被 $primary-hover 部分覆盖
    for k in sorted(palette, key=len, reverse=True):
        template = template.replace(k, palette[k])

    return template