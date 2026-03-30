
# 0x6. 飞书和 cc-connect 详细配置教程

这一节我尽量按“照着抄就能跑”的方式写，把前面踩过的坑和问过的问题都放进来。

## 0x6.1 先准备好这几样东西

- 一台长期在线的本地机器，已经装好 `Codex CLI`
- 你的项目目录，比如我这边是 `/Users/bbuf/工作目录/Common`
- `cc-connect`
- 一个飞书企业自建应用
- 如果你要让 Agent 直接 SSH 本地可达机器，比如 `b200`，那本地机器自己的 SSH、Docker、远端别名也要先配好

## 0x6.2 飞书应用怎么创建

进入飞书开放平台：

```text
https://open.feishu.cn/
```

然后按这个顺序来：

1. 创建“企业自建应用”
2. 打开 `应用能力 -> 机器人`
3. 打开 `事件订阅`，选择 **长连接**
4. 添加事件：

```text
im.message.receive_v1
```

5. 到 `权限管理` 里开权限
6. `创建版本`
7. `发布`

这里最容易漏掉的是最后两步。飞书后台里很多配置“看起来已经勾上了”，但不创建版本、不发布，实际上不会生效。

## 0x6.3 权限到底怎么勾

如果你只想先跑通私聊机器人，我这边实际联通过的一组最小权限可以参考：

- `contact:user.base:readonly`
- `im:message.p2p_msg:readonly`
- `im:message:send_as_bot`

如果你还希望群里 `@机器人` 也能工作，再加这两个：

- `im:message.group_at_msg:readonly`
- `im:message.group_msg`

不需要勾的就别碰，比如门禁、审批、云文档、日历、会议室、多维表格这些都和 `cc-connect` 没关系。

最稳妥的办法是直接在飞书权限搜索框里搜上面的 permission code，一个一个勾。

## 0x6.4 凭据从哪拿

发布完成后，在飞书应用后台拿到：

- `App ID`
- `App Secret`

`App ID` 一般长这样：

```text
cli_xxxxxxxxxxxxxxxx
```

`App Secret` 就是一串密钥字符串。

有个安全提醒很重要：如果你把 `App Secret` 发到聊天记录、截图或者公开文档里，后面最好去飞书后台重新生成一次。

## 0x6.5 config.toml 怎么写

`cc-connect` 的全局配置文件默认在：

```text
~/.cc-connect/config.toml
```

如果你先只挂一个飞书机器人，可以从这份模板开始：

```toml
language = "zh"

[log]
level = "info"

[stream_preview]
enabled = false

[[projects]]
name = "common"
quiet = true

[projects.agent]
type = "codex"

[projects.agent.options]
work_dir = "/Users/bbuf/工作目录/Common"
mode = "yolo"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_xxxxxxxxxxxxxxxx"
app_secret = "your_feishu_app_secret"
progress_style = "compact"
```

如果你想像我一样同一个项目挂两个飞书机器人，那就继续往后追加一段 `[[projects.platforms]]`：

```toml
language = "zh"

[log]
level = "info"

[stream_preview]
enabled = false

[[projects]]
name = "common"
quiet = true

[projects.agent]
type = "codex"

[projects.agent.options]
work_dir = "/Users/bbuf/工作目录/Common"
mode = "yolo"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_first_bot"
app_secret = "first_secret"
progress_style = "compact"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_second_bot"
app_secret = "second_secret"
progress_style = "compact"
```

这几个配置我建议直接按下面理解：

- `work_dir`
  就是 Codex 真正工作的项目目录。
- `mode = "yolo"`
  这项几乎是远端开发刚需。不开的话，联网、SSH、跑 shell、进 Docker 这类动作很容易被沙箱挡住。
- `quiet = true`
  默认少发 thinking 和工具过程。
- `[stream_preview].enabled = false`
  关掉流式半成品预览，飞书里不会刷一长串半成品。
- `progress_style = "compact"`
  进度展示尽量紧凑一点。

## 0x6.6 为什么我这里是 yolo

这个点之前我专门踩过坑。

如果你想让飞书里的 Codex 真的能访问本地机器本来就能访问的目标，比如：

- `ssh b200`
- 进本地 Docker
- 跑 benchmark
- 连外网下载模型或工具

那 `suggest` 基本不够用，得切到：

```toml
mode = "yolo"
```

这是我后面把 `b200` 跑通的关键之一。

代价也很明确：权限更大，所以更要看结果、看 diff、看 benchmark，而不是把它当黑盒。

## 0x6.7 怎么启动 cc-connect

配置写好以后，直接在新终端运行：

```bash
cc-connect -config ~/.cc-connect/config.toml
```

看到它正常启动后，就可以去飞书里私聊机器人测试。

第一轮联调我建议这么做：

1. 私聊机器人发一句普通话
2. 确认它能正常回消息
3. 再让它做一个很轻的本地命令
4. 确认没问题后，再让它去 SSH、跑 benchmark、进 Docker

如果你后面想省掉手动启动，也可以再单独做成 macOS 后台服务，但手工启动是最容易排障的第一步。

## 0x6.8 怎么开始一个新会话，怎么结束当前任务

这几个是飞书里最常用的 slash 命令：

- **`/new`**
  开一个新会话。
- **`/new b200-debug`**
  开新会话并命名。
- **`/stop`**
  停掉当前正在跑的任务。
- **`/list`**
  查看已有会话。
- **`/switch <id>`**
  切到某个旧会话。
- **`/current`**
  看当前会话是谁。
- **`/history 20`**
  回看最近 20 条消息。
- **`/mode yolo`**
  在会话级切权限模式。
- **`/reasoning high`**
  提高推理强度。
- **`/quiet`**
  会话级减少中间过程输出。
- **`/help`**
  查帮助。

如果你想“彻底开始一条新的任务线”，我建议直接连着发：

```text
/stop
/new
```

前者停当前任务，后者切到一条新的上下文。

## 0x6.9 怎么查当前进度

这个我后来反而不太依赖 slash 命令，而是直接发自然语言：

```text
总结一下当前进度、已经完成的修改、正在跑的验证，以及下一步计划。
```

因为这类问题本质上是“让模型基于上下文做摘要”，自然语言往往比专门记命令更顺手。

## 0x6.10 token 用量和剩余额度怎么看

这一点我也专门问过。

结论是：在我这条 `cc-connect + 飞书 + Codex` 链路里，我目前没有验证到一个稳定可用的官方 slash 命令去看 token 用量或剩余额度。

所以别先入为主去记一个假的：

```text
/usage
/quota
```

至少在我这套配置里，我没有把它们当成稳定入口来依赖。

## 0x6.11 怎么让飞书里别刷一堆 Bash 和工具日志

这个也是实战里特别重要的一点。

如果你不做任何收敛，飞书里会刷大量：

- `工具 #35: Bash`
- `rg`
- `sed`
- thinking 过程

手机上几乎没法看。

我最后用的是这三个配置：

```toml
quiet = true

[stream_preview]
enabled = false

progress_style = "compact"
```

配完之后，飞书里就更像“结果面板”，不是“终端镜像”。

## 0x6.12 一个项目挂多个飞书机器人怎么配

这个其实非常简单，不用复制两套项目。

同一个 `[[projects]]` 下面，多写几段：

```toml
[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_bot_a"
app_secret = "secret_a"
progress_style = "compact"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_bot_b"
app_secret = "secret_b"
progress_style = "compact"
```

就可以让两个飞书机器人同时指向同一个项目目录、同一个 Codex agent。

这个特别适合：

- 一个机器人自己用，一个给搭档用
- 一个当稳定入口，一个当实验入口
- 一个跑主任务，一个专门跑调试

## 0x6.13 项目目录下的 AGENTS.md 可以顺手补上

如果你希望 Agent 在这个项目里天然知道怎么用 `cc-connect` 回消息、加 cron，也可以在项目根目录放一个 `AGENTS.md`。

我这边现在放的是这种思路：

```md
# cc-connect Integration

## Scheduled tasks (cron)
cc-connect cron add ...

## Send Message To Current Chat
cc-connect send -m "short message"
```

它不是飞书接入的硬前置条件，但对长期用这套链路的人很有帮助。

## 0x6.14 最后再提醒几个容易忘的坑

- 权限和事件勾完以后，一定要 `创建版本` + `发布`
- 如果飞书里发出来的过程信息太多，优先检查 `quiet`、`stream_preview`、`progress_style`
- 如果机器人连得上但 SSH 不通，先看是不是还在 `suggest`，不是 `yolo`
- 如果你把 `App Secret` 发到聊天、截图、公开文档里，记得后面重置
- 私聊能通、群里不通，优先回去检查群聊相关权限和应用发布状态
