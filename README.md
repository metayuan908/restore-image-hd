# Restore Image HD · GPT 高清重绘与拼接

通过语义蒙版、全局构图锚点和原生细节分块重绘，将图片重建为 HD、2K、4K、8K 或自定义尺寸。

作者：Yuanwei  
版本：0.4.1+codex.20260731052910

## 内容

- `skills/restore-image-hd/`：风险预检、蒙版局部重绘、构图锚点、分块重绘、色彩协调、拼接及交付脚本。
- `skills/codex-gpt-image/`：通过当前机器的 Codex 登录状态调用 GPT Image 的配套 CLI。
- `.codex-plugin/plugin.json`：Codex 插件清单。

## 环境与安装

需要 Python 3.10+，以及当前机器可用的 Codex 登录状态。

```bash
python3 -m pip install -r requirements.txt
```

作为独立技能使用时，将 `skills/` 下的两个目录一起复制到 `~/.codex/skills/`，保持两个技能为相邻目录。插件包结构保留在本仓库中。

凭据仅从使用者本机读取；仓库不包含登录文件或 API 密钥。缺少登录状态时请运行 `codex login`。实际模型可用性取决于账号权限及服务端支持。

## 使用

在 Codex 中调用 `$restore-image-hd`，提供原图及目标尺寸。详细工作流见 [技能说明](skills/restore-image-hd/SKILL.md)。

命令行先检查重绘路线：

```bash
python3 skills/restore-image-hd/scripts/plan_redraw_route.py --help
python3 skills/restore-image-hd/scripts/gpt_coherent_redraw.py --help
python3 skills/restore-image-hd/scripts/gpt_segmented_redraw.py --help
```

产品、人物、建筑等有连续结构的画面应使用带全局锚点或语义蒙版的路线。包含文字、Logo、二维码的图片需要按技能说明保护或确定性合成这些元素。

## 分辨率说明

交付图是 GPT 重绘并拼接的目标尺寸画布，不应称为单张模型原生 4K/8K。请保留每次原生生成尺寸、分块及拼接记录。脚本运行会发起图像生成请求，请先查看 dry-run 和风险预检结果。

本仓库从本地安装插件导出，移除了 Python 缓存，保留原始技能与脚本。未附加开源许可证。
