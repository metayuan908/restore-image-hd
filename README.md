# Restore Image HD · GPT 高清重绘与拼接

通过语义蒙版、全局构图锚点和原生细节分块重绘，将图片重建为 HD、2K、4K、8K 或自定义尺寸。

作者：Yuanwei  
版本：0.4.2+codex.20261029

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

## v0.4.2：稳定性与耗时优化

- 根据实际返回尺寸记录容量；默认交付裁块预算为 120 万像素，包含上下文，不假定请求尺寸一定实现。
- 生成输入固定，默认最多 2 路并发；遇到限流降为串行，合成始终按区域优先级执行。
- 原生文件保留不变；逐块记录输入哈希、状态与耗时，续跑只复用匹配结果。
- 明确请求不透明细节图，拒绝异常透明结果，原始透明通道单独恢复。
- 在颜色匹配前进行受限几何检查；大幅错位阻止导出，并支持仅重绘失败区域。
- 支持真实重叠区的低差异接缝选择；复杂重叠回退会明确记录，仍需视觉检查。

分区流程可用 `--stage prepare|generate|compose|all` 分开执行。试绘后检查 `pilot-composite.png`，再使用 `--resume --approve-pilot`。失败区域使用 `--resume --retry-region NAME`。可选的 `--alignment affine` 需要另行安装 `opencv-python-headless`；默认平移校正不需要 OpenCV。

离线验证（不读取登录凭据、不发送生图请求）：

```bash
python3 -m unittest discover -s tests -v
```

21 项离线测试通过；真实服务端并发加速尚未实测，不据模拟测试承诺任务耗时。
