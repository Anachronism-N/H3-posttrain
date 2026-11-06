# Codex CLI 安装与 Idealab AK 配置指南

核实日期：2026-10-21。以下命令面向 Linux / macOS 的 Bash 终端。

> 安全提示：本文不包含任何 AK。使用时通过隐藏输入把 AK 加载到环境变量；不要将 AK 写入本文、项目配置、Git 仓库，也不要直接写进命令行参数（多用户机器上其他用户可通过 `ps` 看到命令行）。

## 实测结论：已走通

在本机（Linux x64）用官方 Codex CLI 0.163.0 连接 Idealab，实测通过：

| 项目 | 值 |
| --- | --- |
| Base URL | `https://idealab.alibaba-inc.com/api/openai/v1` |
| 协议 | OpenAI Responses API（`wire_api = "responses"`） |
| 推荐模型 | `gpt-5.6-terra`（两枚 AK 均有权限） |
| 认证 | `Authorization: Bearer <AK>`，由环境变量 `IDEALAB_API_KEY` 提供 |

已验证：

- `codex exec` 纯文本回复：通过（两枚 AK 各验证一次）。
- 工具调用：Codex 在只读沙箱中执行 `cat ./probe.txt`，并原样返回文件中的随机令牌：通过。
- 直接使用本机 `~/.codex/config.toml`（见第 2 节），不加任何 `-c` 覆盖参数：通过。

模型权限实测（最小 Responses 请求；AK 按对话中给出的顺序编号）：

| 模型 | 第一枚 AK | 第二枚 AK |
| --- | --- | --- |
| `gpt-5.6-terra` | 可用 | 可用 |
| `gpt-5.6-luna` | 可用 | 403 无权限（CE-011） |
| `gpt-5.6-sol` | 403 无权限（CE-011） | 403 无权限（CE-011） |
| `gpt-5.6` | 400 模型名错误（CE-001） | 400 模型名错误（CE-001） |
| `gpt-5.4` | 400 模型名错误（CE-001） | 400 模型名错误（CE-001） |
| `Qwen3.6-Plus-DogFooding` | 403 无权限（CE-011） | 403 无权限（CE-011） |
| `qwen3-coder-plus` | 404 模型已下架 | 未测试 |

注意：

- GPT-5.6 必须写带后缀的完整型号名（`-terra` / `-luna` / `-sol`），与 Codex 0.163.0 内置模型元数据中的名称一致；裸的 `gpt-5.6` 会被 Idealab 拒绝。
- `GET /models` 只返回了已下架的 `qwen3-coder-plus`，不能作为可用模型列表。

## 1. 安装 Codex CLI

**方式一（已有 Node.js 和 npm）**：

```bash
npm install -g @openai/codex@latest
codex --version
```

**方式二（本机实际采用，不依赖 Node.js）**：从官方 npm registry 下载 Linux x64 平台包 `@openai/codex@0.163.0-linux-x64`，按 registry 元数据中的 SHA-512 校验通过后解压：

- 可执行文件：`~/.local/share/codex-cli/0.163.0/package/vendor/x86_64-unknown-linux-musl/bin/codex`
- 软链接：`~/.local/bin/codex`（该目录已在 `PATH` 中）

```bash
codex --version   # 输出 codex-cli 0.163.0
command -v codex  # 输出 ~/.local/bin/codex
```

如果以后改用 npm 管理 Codex，先删除 `~/.local/bin/codex` 软链接，避免两个 `codex` 冲突。

## 2. Codex 配置

默认用户配置文件是 `~/.codex/config.toml`；如果设置了 `CODEX_HOME`，则使用该目录下的 `config.toml`。本机已写入以下内容（不含 AK）：

```toml
model_provider = "idealab"
model = "gpt-5.6-terra"

[model_providers.idealab]
name = "Idealab"
base_url = "https://idealab.alibaba-inc.com/api/openai/v1"
env_key = "IDEALAB_API_KEY"
wire_api = "responses"
requires_openai_auth = false
```

说明：

- `env_key` 填的是环境变量的**名称**，不是 AK 本身。
- `model` 和 `model_provider` 是顶层键，必须放在任何 `[表名]` 之前；修改已有配置时要合并，不要重复定义相同的键或表。
- 当前 Codex 只支持 `wire_api = "responses"`，旧教程中的 `wire_api = "chat"` 已不可用。
- 这种自定义 provider 使用环境变量认证，不需要 `codex login`，也不需要把 AK 写入 `auth.json`。
- 如需改用 `gpt-5.6-luna`，只改 `model` 即可；但它仅第一枚 AK 有权限。

## 3. 设置 Idealab AK

在准备运行 Codex 的同一个 Bash 终端执行，粘贴任意一枚 AK 后回车：

```bash
read -r -s -p '请输入 Idealab AK（输入不会显示）: ' IDEALAB_API_KEY
printf '\n'
export IDEALAB_API_KEY
```

检查是否已设置（只输出状态，不输出密钥）：

```bash
if [ -n "${IDEALAB_API_KEY:-}" ]; then
  printf 'IDEALAB_API_KEY 已设置\n'
else
  printf 'IDEALAB_API_KEY 未设置\n'
fi
```

环境变量只对当前终端及其子进程生效，新开终端需要重新输入。不要用 `echo "$IDEALAB_API_KEY"`、`printenv` 或 shell 调试追踪输出密钥，也不要明文写进 `.bashrc`。Codex 默认会从它执行的 shell 命令环境中剔除名称含 `KEY`、`SECRET`、`TOKEN` 的变量，模型运行的命令看不到这个 AK。

## 4. 启动与验证

在目标项目目录启动交互界面：

```bash
codex
```

最小连接测试（只读沙箱、不保存会话记录；会产生少量调用费用）：

```bash
codex exec --ephemeral --sandbox read-only '只回复“连接成功”，不要读取项目文件、执行命令或修改文件。'
```

需要绕开 Codex、直接确认 AK 和模型名时，可以用下面的请求。AK 通过标准输入传给 curl，不会出现在命令行参数中：

```bash
curl -sS -K - https://idealab.alibaba-inc.com/api/openai/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.6-terra","input":"只回复 OK","max_output_tokens":200,"store":false}' \
  <<< "header = \"Authorization: Bearer ${IDEALAB_API_KEY}\""
```

使用结束后可清除当前终端中的 AK：

```bash
unset IDEALAB_API_KEY
```

## 5. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `codex: command not found` | 确认 `~/.local/bin` 在 `PATH` 中，或 npm 全局 `bin` 目录在 `PATH` 中。 |
| 提示缺少 `IDEALAB_API_KEY` | 没有在启动 Codex 的同一个终端中输入并 `export` AK。 |
| `用户设置的model错误`（CE-001） | 模型名不存在，例如裸的 `gpt-5.6`；改用完整型号名，如 `gpt-5.6-terra`。 |
| `您没有当前模型的权限`（CE-011） | 该 AK 没有这个模型的权限；换一个模型或另一枚 AK，或按报错中的链接申请权限。 |
| `模型已下架` | 模型已经停用，换用当前可用的模型。 |
| `stream disconnected before completion` | 实测在模型名无效时会出现这个报错，网关的真实错误信息没有透传出来；用第 4 节的 curl 请求查看真实错误。 |
| `Model metadata for ... not found` | Codex 不认识这个模型名，通常说明名称写错了；使用 `gpt-5.6-terra` 时不会出现。 |
| `401 Unauthorized` | AK 无效或粘贴时带了多余空格。 |
| 超时或无法连接 | 检查阿里内网 / VPN、DNS、代理和证书；不要通过关闭 TLS 校验绕过证书问题。 |
| 仍提示 OpenAI / ChatGPT 登录 | 没有加载到正确的配置文件，或当前 provider 不是 `idealab`。 |

## 6. 参考来源

- [官方 npm 包：@openai/codex](https://www.npmjs.com/package/@openai/codex)
- [Codex CLI 官方说明](https://developers.openai.com/codex/cli)
- [Codex 配置参考](https://developers.openai.com/codex/config-reference)
- [Codex 高级配置](https://developers.openai.com/codex/config-file/config-advanced)
- [Codex 配置样例：仅支持 Responses](https://developers.openai.com/codex/config-file/config-sample)
- [Idealab 官方文档入口（需内部登录）](https://idealab.alibaba-inc.com/#/open/docs/model/overview)

以上结论均为本机实测：官方安装包按 npm registry 元数据做了 SHA-512 校验，CLI 参数以已安装 `codex exec --help` 的输出为准。Idealab 登录后的文档正文尚未读取；模型权限可能随时间变化，以实际请求结果为准。
