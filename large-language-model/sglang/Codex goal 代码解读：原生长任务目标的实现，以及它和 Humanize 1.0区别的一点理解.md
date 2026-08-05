> 写在前面：本文对 Codex `/goal` 的源码（rust）阅读和分析由 Claude Opus 4.8 完成。我做的事情是给它指阅读方向、询问我感兴趣的细节（状态怎么存、续跑怎么触发、完成怎么判定、和 Humanize 1.0 细节的对比），并对成文做了一些 refine。代码引用都来自 `openai/codex` 仓库，可自行核对。对比Humanize 1.0部分有我自己的一些理解，不保证100%正确注意鉴别。

# 0x0. 太长不看版

- goal 被实现成一个挂在 Agent 事件循环上的 extension，靠线程、turn、token、工具这些生命周期回调驱动，全程没有外部进程，也没有 bash 中转。
- 目标状态存在 SQLite 的 `thread_goals` 表里，一个 goal 一行，有 6 种状态；token 用量和耗时也记在这行上，并发更新做了保护，同一笔用量不会重复扣或算错。
- 续跑是事件驱动的：线程空闲才触发，还要过几道安全闸（Plan 模式、Review 子代理、有用户输入排队都不续）。续跑靠往活线程注入一段"转向"上下文，不重开进程，所以它的 continuation prompt 只有 50 行，而 Humanize 1.0 每轮得重读文件把上下文灌回去。
- 完成判定是同一个模型在同一线程里自审，靠 continuation.md 那段把举证责任压到"完成"侧的 prompt。这是它和 Humanize 1.0 最大的差别：Humanize 1.0 另起一个独立 Codex 进程做对抗式审查，吐 `COMPLETE` 才放行。
- 工具契约在代码和 schema 两层硬限权限：模型只能 create / get / update(complete|blocked)，pause/resume/clear 走 app-server，归用户和系统。
- 预算只按 token 卡停（撞线由一条原子 SQL 翻成 `budget_limited`），wall-clock 只记不卡；token 记账还减掉了命中缓存的部分。
- 出错有熔断：不可重试的错误（尤其 compaction error）直接把 goal 打成 `Blocked`，防止空闲续跑在同一个错误上无限烧 token。
- `/goal` 管的是 Codex 进程里的续跑、记账和状态流转；Humanize 1.0 管的是另一件事：让外部审查员来决定能不能收工，并把每轮记录留在文件里。后面这两点，`/goal` 目前没有。

# 0x1. 前言

Humanize 1.0 主要靠 RLCR（Ralph-Loop with Codex Review）运转，大体是一套外挂的 bash 状态机：用 Stop hook 拦住退出，把状态落到 `.humanize/` 的 markdown 里，再另起一个独立的 Codex 进程做审查，不通过就继续下一轮。它解决的问题很明确：让"完成"变成一个外部判定，而不是 Agent 自己说完就完。

后来 Codex 自己上了 `/goal`，文档把它定义成"长时间工作的持久目标"。当时我的第一反应是：这不就是把 Humanize 1.0 干的事内置了一份吗？所以这次我直接把 `openai/codex` 拉下来，把 `/goal` 的实现从头读了一遍（commit `04483f4`，goal 相关代码集中在 `codex-rs/ext/goal/`）。读完之后我改了看法：`/goal` 主要处理 Codex 运行时里的续跑和状态，Humanize 1.0 更像一套外部审查流程。同样都在谈"长任务"，落点其实不一样。这篇就把代码摊开，看看 `/goal` 怎么实现，以及它和 Humanize 1.0 到底差在哪。

先给一个结论性的对比，后面再展开：

| 职责 | Humanize 1.0 放在哪 | Codex `/goal` 放在哪 |
| --- | --- | --- |
| 持久目标状态 | `.humanize/rlcr/<ts>/` 一堆 markdown | 数据库里的一行（SQLite） |
| 继续推进的触发 | bash Stop hook + Ralph loop 外部驱动 | 线程一空闲，进程自己续上 |
| 完成判定 | 另起独立 Codex 进程审查，吐 `COMPLETE` 放行 | 同一个模型自审，自己标完成 |
| 预算 / 停止 | 最大轮数（默认 42 轮） | 按 token 预算停，wall-clock 只记不卡 |
| 生命周期权限 | 全在用户和脚本手里 | 模型只能标完成/卡住，其余归用户 |

cookbook 里这张图讲的是 `/goal` 的基本模型：

![图 1：Goal 把一次性对话变成带证据检查的持续推进循环](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/88f846e6-4057-4214-9cbb-7844608e8be1.png)

*图 1. Goal 把一次性对话变成带证据检查的持续推进循环（图自 OpenAI Cookbook）。*

普通 prompt 是"请求 → 工作 → 结果 → 等待"，给一条指令、做一件事就停；goal 把它变成"工作 → 检查 → 继续或完成"的循环。区别在最后那个分叉：这一轮要不要停，由证据检查决定，而不是模型觉得差不多了就行。后面几节就顺着代码看这套循环具体怎么搭起来。

# 0x2. 整体结构：goal 作为 extension 接入事件循环

`/goal` 的接入方式不太直观：它是一个挂在 Agent 事件循环上的 extension，而不是 TUI 里一个 slash command 的处理函数。`codex-rs/ext/goal/src/extension.rs` 里的 `GoalExtension` 实现了一整排 trait：

```rust
impl<C> ThreadLifecycleContributor<C> for GoalExtension<C> { ... }  // 线程开始/恢复/空闲/停止
impl<C> TurnLifecycleContributor   for GoalExtension<C> { ... }     // 每个 turn 开始/结束/中断/出错
impl<C> TokenUsageContributor      for GoalExtension<C> { ... }     // 每次 token 用量回调
impl<C> ToolLifecycleContributor   for GoalExtension<C> { ... }     // 每个工具调用结束
impl<C> ToolContributor            for GoalExtension<C> { ... }     // 向模型暴露 goal 工具
impl<C> ConfigContributor          for GoalExtension<C> { ... }     // 配置热更新
```

接入点不一样，能管的范围就不一样。Humanize 1.0 站在 Codex 外面，用 hook 在它退出那一刻插一脚；`/goal` 是 Codex 里面的一个观察者，线程每一次状态跃迁它都在现场：开一个 turn、烧一批 token、跑完一个工具、空闲下来。整个 goal 的逻辑就是这几个回调拼出来的，没有外部进程，没有 bash，没有文件中转。

cookbook 用一张图概括了 goal 给一个线程加了哪几样东西：

![图 2：Goal 为当前线程加入持久状态、继续推进能力、控制入口和证据检查](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-2/875a58a8-1948-4e89-ab65-453233d7b608.png)

*图 2. Goal 为当前线程加入持久状态、继续推进能力、控制入口和证据检查（图自 OpenAI Cookbook）。*

这四样正好对应后面几节：持久状态是 0x3 的 `thread_goals` 表，继续推进是 0x4 的空闲续跑，控制入口是 0x5 的工具契约和 app-server，证据检查是 0x6/0x7 的完成审计。它们都挂在同一个线程上，这也是它和 Humanize 1.0 那套外部文件状态最不一样的地方。

# 0x3. 状态模型：thread_goals 表与生命周期

Humanize 1.0 的状态是一坨 markdown：`state.md`、`goal-tracker.md`、`round-N-summary.md`、`round-N-review-result.md`……好处是人能直接 `cat`、能 git diff、能事后复盘；坏处是它是文本，没有事务，靠脚本去解析和校验。

`/goal` 的状态是 `codex-rs/state/src/model/thread_goal.rs` 里的一个结构体，落到 SQLite 的 `thread_goals` 表，按 `thread_id` 作用域：

```rust
pub struct ThreadGoal {
    pub thread_id: ThreadId,
    pub goal_id: String,
    pub objective: String,
    pub status: ThreadGoalStatus,
    pub token_budget: Option<i64>,
    pub tokens_used: i64,
    pub time_used_seconds: i64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

有两个细节。

一个是生命周期，6 个状态，比 cookbook 讲的多两个：

```rust
pub enum ThreadGoalStatus {
    Active,         // 活跃，可以自动继续
    Paused,         // 用户暂停
    Blocked,        // 模型自己判定卡死
    UsageLimited,   // 撞到账号用量上限
    BudgetLimited,  // 撞到本 goal 的 token 预算
    Complete,       // 完成
}
```

cookbook 里只讲了 active/paused/complete/budget-limited，实际代码里多了 `Blocked` 和 `UsageLimited`，这两个状态对应的正好是长任务里很容易遇到的情况：模型钻死胡同，或者账号额度耗尽。后面会看到它们都有专门的兜底逻辑。`is_terminal()` 只认 `BudgetLimited | Complete`，也就是说 `Blocked`、`Paused`、`UsageLimited` 都不是终点，都还能被恢复。

另一个是 token 和时间，它俩是表里的一等字段：`tokens_used`、`time_used_seconds`、`token_budget` 直接进 schema，不是事后从日志里捞。这就让预算判定可以做成一条原子 SQL（见 0x6），而不是 Humanize 1.0 那种"数到第几轮了"的近似。Humanize 1.0 的"预算"是 `--max N`，默认 42 轮，其实数的是迭代次数；`/goal` 这边记的是真实 token 数和真实耗时（wall-clock），其中只有 token 能设硬上限（怎么设、怎么影响停止，见 0x6）。一个是"你还能折腾几次"，一个是"你还能烧多少钱"，后者对长任务更贴近实际成本。

# 0x4. 续跑机制：空闲事件触发与转向注入

这块是 `/goal` 和 Humanize 1.0 区别最大的地方。

Humanize 1.0 怎么继续下一轮？Stop hook 在 Codex 退出时触发，脚本拉起 `codex exec` 审查 summary、`codex review --base` 审查代码，没问题就写一个 next-round prompt，重新拉起一个 Codex 进程跑下一轮。每一轮都是一次冷启动。

`/goal` 完全不是这个路子。继续推进的入口是线程空闲事件，`extension.rs`：

```rust
fn on_thread_idle<'a>(&'a self, input: ThreadIdleInput<'a>) -> ExtensionFuture<'a, ()> {
    Box::pin(async move {
        let Some(runtime) = goal_runtime_handle(input.thread_store) else { return; };
        if let Err(err) = runtime.continue_if_idle().await {
            tracing::warn!("failed to continue active goal for idle thread {}: {err}", ...);
        }
    })
}
```

而这个 idle 事件本身是有门槛的，`core/src/tasks/lifecycle.rs`：

```rust
pub(crate) async fn emit_thread_idle_lifecycle_if_idle(&self) {
    if self.active_turn.lock().await.is_some()
        || self.input_queue.has_trigger_turn_mailbox_items().await
    {
        return;   // 还有 turn 在跑、或者用户输入在排队，就不发 idle 事件
    }
    // ...才会通知 goal extension
}
```

也就是说，只有当前没有 turn 在跑、用户也没有排队输入时，才会触发一次"要不要继续"的判断。cookbook 说的"只有线程空闲、Goal 活跃且预算充足时才会继续"，落到代码里就是这两个 if。cookbook 把这个调度条件画成了一张图：

![图 3：只有当 Goal 活跃、线程空闲且没有用户输入排队时，Codex 才会继续](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/releases/download/mdnice-assets-2026-08-05-1/4ffc7f52-9892-4448-9244-4c5d3109de92.png)

*图 3. 只有当 Goal 活跃、线程空闲且没有用户输入排队时，Codex 才会继续（图自 OpenAI Cookbook）。*

图里这几个判断条件，落到代码就是上面 `emit_thread_idle_lifecycle_if_idle` 的两个 if，加上下面 `try_start_turn_if_idle` 的再次复查。续跑不是一个定时循环在转，而是这些条件都满足、线程真正空下来时，才放一次行。

`continue_if_idle` 自己也很克制（`ext/goal/src/runtime.rs`）。它先抢一把 `goal_state_lock` 信号量（防止读到 goal 之后、启动 turn 之前，用户在另一条路径上把 goal 改了），确认 goal 还是 `Active`，然后干这件事：

```rust
let item = continuation_steering_item(&protocol_goal_from_state(goal));
if let Err(err) = thread.try_start_turn_if_idle(vec![item]).await {
    let reason = err.reason();
    tracing::debug!(?reason, "skipping goal continuation because automatic idle work was rejected");
}
```

注意 `try_start_turn_if_idle`，这是宿主层的安全闸（`core/src/session/inject.rs`），它会再查一遍：

```rust
if self.input_queue.has_trigger_turn_mailbox_items().await { return Err(PendingTriggerTurn); }
if self.collaboration_mode().await.mode == ModeKind::Plan { return Err(PlanMode); }
// active_turn 已存在 → Busy
// 启动前再复查一次 mailbox、Plan mode、reservation 有没有被别人抢走
```

三道检查，而且关键条件查了不止一次（防 race）。`PlanMode` 这条尤其值得说：Plan 模式下自动 idle work 直接被拒。这对应 cookbook 那句"只有计划、没有实际执行的工作不会触发继续"。`on_turn_start` 里只要发现是 Plan 模式，就把当前 turn 的 active goal 清掉，这一轮不算 goal 进度。

再看继续推进的形态：它把一段"转向"注入到一个新 turn 里，既不用重新拉进程，也不用塞一条 user 消息。`ext/goal/src/steering.rs`：

```rust
fn goal_context_input_item(prompt: String) -> ResponseItem {
    ContextualUserFragment::into(InternalModelContextFragment::new(
        InternalContextSource::from_static("goal"),   // 来源打标为 "goal"
        prompt,
    ))
}
```

继续的 prompt 是一个 source 标成 `goal` 的内部上下文片段，直接插进活着的线程。线程没死，之前看过的文件、跑过的命令、产生的 diff、推理轨迹全都还在上下文里。

这也解释了一个现象：为什么 Humanize 1.0 的 continuation prompt 那么长，`/goal` 的那么短？

把两边的"继续"prompt 放一起看就懂了。Humanize 1.0 的 `next-round-prompt.md` 开头是这样：

```text
## Round Re-anchor (REQUIRED FIRST STEP)
Before writing code:
- Re-read @{{PLAN_FILE}}
- Re-read @{{GOAL_TRACKER_FILE}}
- Re-read the most recent round summaries/reviews ...
- Write the current round contract to @{{ROUND_CONTRACT_FILE}}
```

它必须在每一轮开头让模型重新读 plan、读 goal-tracker、读上一轮 summary，因为每一轮都是新进程，上下文是从文件里重建的。Re-anchor、round contract、task lane（mainline/blocking/queued 三种 tag）、bitlesson 选择……这一大套脚手架，主要是在"重新把记忆灌回模型脑子里"。

`/goal` 不需要重建，因为线程一直活着。所以它的 `continuation.md` 第一句就是：

```text
Continue working toward the active thread goal.
...
Use the current worktree and external state as authoritative. Previous conversation
context can help locate relevant work, but inspect the current state before relying on it.
```

一句"接着干"，加一句"以当前 worktree 和外部状态为准"。两边 prompt 长度差这么多，根子在架构对上下文的处理：一个靠文件重建记忆，一个靠线程保留记忆。

最后是出错兜底。`on_turn_error` 里有一段专门防止自动续跑空转的处理：

```rust
let reason = match input.error {
    CodexErrorInfo::UsageLimitExceeded => ActiveGoalStopReason::UsageLimit,
    // The turn has ended because the error was non-retryable or its retries were
    // exhausted. Block the goal to prevent automatic continuation from looping and
    // consuming tokens, as can happen with compaction errors.
    _ => ActiveGoalStopReason::TurnError,   // → 把 goal 置成 Blocked
};
```

turn 出了不可重试的错（注释点名了 compaction error），就把 goal 打成 `Blocked`，免得 idle 事件一次次把它拉起来、在同一个错误上空转烧 token。这是 Humanize 1.0 那种"外部数轮数"很难精确做到的：它得等到下一轮 review 才发现不对，中间可能已经空转好几次了。

# 0x5. 工具契约与生命周期权限

cookbook 里提到工具契约会限制生命周期权限。读代码会发现这个限制是实打实的，代码和参数 schema 两层都做了硬约束。

模型能看到的 goal 工具只有三个（`ext/goal/src/spec.rs`）：`get_goal`、`create_goal`、`update_goal`。其中 `create_goal` 的 description 第一句就是：

```text
Create a goal only when explicitly requested by the user or system/developer
instructions; do not infer goals from ordinary tasks.
```

明确禁止模型从普通任务里"脑补"出一个 goal。`update_goal` 的参数 schema 更狠，status 是个只有两个值的枚举：

```rust
JsonSchema::string_enum(
    vec![json!("complete"), json!("blocked")],
    Some("Set to `complete` only when the objective is achieved ...
          Set to `blocked` only after the same blocking condition has recurred for
          at least three consecutive goal turns ...")
)
```

模型连"暂停/恢复/预算受限"这些状态名都填不进去。万一它硬塞，`tool.rs` 的 `handle_update` 还有第二道防线：

```rust
if !matches!(args.status, ThreadGoalStatus::Complete | ThreadGoalStatus::Blocked) {
    return Err(FunctionCallError::RespondToModel(
        "update_goal can only mark the existing goal complete or blocked; pause, resume,
         budget-limited, and usage-limited status changes are controlled by the user or system"
    ));
}
```

那 pause/resume/clear/edit 走哪？走完全不同的一条路，app-server 协议。`app-server/src/request_processors/thread_goal_processor.rs` 里是 `thread_goal_set` / `thread_goal_clear`，TUI 的 `/goal pause`、`/goal resume`、`/goal clear`、`/goal edit` 都从这里进，最后调到 runtime 的 `apply_external_goal_set` / `apply_external_goal_clear`。

这是一个挺干净的权限分离：

- 模型在线程内：`create`（仅当被明确要求）/ `get` / `update(complete|blocked)`。
- 用户/系统在带外：set / edit / pause / resume / clear，外加系统触发的 budget-limited / usage-limited。

而且 `create_goal` 还限制了"一个线程同时只能有一个未完成的 goal"：

```rust
.ok_or_else(|| FunctionCallError::RespondToModel(
    "cannot create a new goal because this thread has an unfinished goal;
     complete the existing goal first"))?
```

Humanize 1.0 的 plan/acceptance criteria 也是用户定的、Round 0 之后不让改（goal-tracker 的 immutable section），思路一致。但 Humanize 1.0 是用 hook 校验文件没被偷改来兜底（比如 `goal-tracker-modification.md` 这种 block 模板），属于事后抓；`/goal` 是从工具能力上就不给模型这个权限，属于事前堵。

# 0x6. 预算记账与停止判定

`ext/goal/src/accounting.rs` 把"长任务怎么公平地算钱"这件事抠得很细。

先看 token delta 的算法（`goal_token_delta_for_usage`）：

```rust
pub(crate) fn goal_token_delta_for_usage(usage: &TokenUsage) -> i64 {
    usage.input_tokens
        .saturating_sub(usage.cached_input_tokens)   // 命中缓存的 input 不算钱
        .saturating_add(usage.output_tokens.max(0))
}
```

命中 prompt cache 的 input token 不计入预算，只算未命中的 input 加 output。长任务里上下文越滚越大，如果把缓存的 input 也算进去，预算会被"重复读同一段上下文"飞快吃光，所以这个减法很重要。

再看状态翻转，是一条原子 SQL（`state/src/runtime/goals.rs` 的 `account_thread_goal_usage`）。每次记账，在累加用量的同一条 UPDATE 里顺手判断要不要翻 `BudgetLimited`：

```sql
UPDATE thread_goals SET
    time_used_seconds = time_used_seconds + ?,
    tokens_used       = tokens_used + ?,
    status = CASE
        WHEN status = 'active'
         AND token_budget IS NOT NULL
         AND tokens_used + ? >= token_budget
        THEN 'budget_limited'
        ELSE status
    END,
    ...
```

这里的"翻状态"，指的就是把 goal 的 `status` 从 `active` 改成 `budget_limited`。应用层的笨办法是 read-modify-write：先 `SELECT` 出 `tokens_used`、在代码里加 delta、判断超没超预算、再写回去。`/goal` 没走这条，把累加和判定塞进同一条 `UPDATE`，由数据库一次性执行。为什么非得这样？因为续跑是事件驱动的，`on_tool_finish`、`on_turn_stop` 这些回调可能并发触发记账，read-modify-write 中间那个窗口会让两个回调都读到旧值，结果要么重复扣，要么该翻的状态漏翻。一条原子 `UPDATE` 把读、算、翻焊在一起，窗口就没了。它和外面那把 `progress_accounting_lock` 信号量是两层防护：信号量管应用侧，保证同一笔 delta 不被算两次；原子 SQL 管数据库侧，保证累加和翻状态不被撕开。两层叠起来，记账才是 race-free 的。

顺便把 wall-clock 这条线的作用说清楚，因为容易看错。注意上面这条 SQL：`time_used_seconds` 和 `tokens_used` 都在累加，但 `CASE` 翻 `budget_limited` 只看 `tokens_used + ? >= token_budget`，时间没进判定。wall-clock（`time_used_seconds`，由 `Instant::now()` 的差值累加）是一条只记不卡的线，`ThreadGoal` 里有 `token_budget`，却没有对应的时间预算字段，时间烧多久都不会触发停止。还有个细节，goal 空闲挂着等下一次续跑的那段时间也照算（`account_idle_goal_progress` 里 token delta 记 0、时间照记），所以它反映的是 goal 从创建到现在真实挂了多久，不只是模型在算的那点时间。这个数只用在两处汇报：撞 token 预算、往 turn 里注入 `budget_limit.md` 时，prompt 会带上 `Time spent pursuing goal: N seconds`，让模型知道已经花了多久；goal 标 `complete` 时，`update_goal` 的返回里会带一段提示，让模型把总耗时用人话讲给用户。所以 wall-clock 影响的是"汇报"，不是"停不停"，真正卡停止的只有 token 预算。

最后是撞预算之后怎么办。`on_tool_finish` 里，一旦某次记账把 goal 推过预算线，会当场往正在跑的 turn 里注入一条"预算到了"的转向：

```rust
if goal.status != ThreadGoalStatus::BudgetLimited { return; }
if !runtime.accounting_state().mark_budget_limit_reported_if_new(progress.goal_id.as_str()) {
    return;   // 同一个 goal 只播报一次，不重复打扰
}
let item = budget_limit_steering_item(&goal);
runtime.inject_active_turn_steering(item).await;
```

注入的 `budget_limit.md` 内容是："系统已经把 goal 标成 budget_limited，别开新活了，赶紧收尾：总结进度、列出剩余工作和阻塞、给用户一个明确的下一步。" 它特意强调撞预算不等于完成：`Do not call update_goal unless the goal is actually complete.` 这对应 cookbook 说的"达到预算上限不等于完成目标"，模型不能因为没钱了就把活儿标成 done。

Humanize 1.0 这边对应的是"跑满 42 轮就停"，相对粗一些，而且它没有"算 token"的概念，成本控制的单位是"轮"，不是"钱"。

# 0x7. continuation.md：行为约束与完成判定

前面那些 Rust 代码是骨架。真正管模型怎么继续往下跑的，是 `prompts/templates/goals/continuation.md` 这 50 行 prompt。里面最用力防的，其实就两件事。

一个是偷偷缩小目标（reward hacking by narrowing）。模型在长任务里有个坏习惯，把"难的大目标"悄悄换成"容易过测试的小目标"，然后宣布胜利。这个 prompt 用一整段去堵：

```text
Fidelity:
- Optimize each turn for movement toward the requested end state, not for the
  smallest stable-looking subset or easiest passing change.
- Do not substitute a narrower, safer, smaller, merely compatible, or
  easier-to-test solution because it is more likely to pass current tests.
- Treat alignment as movement toward the requested end state. An edit is aligned
  only if it makes the requested final state more true ...
```

翻译成人话：不许为了"让测试变绿"去换一个更小更安全的解法。它防的是目标被悄悄改小。

另一个是过早宣布完成，Humanize 1.0 做 RLCR，很大一部分也是为了压住这个问题。`/goal` 用一段"完成审计"来对付：

```text
Completion audit:
Before deciding that the goal is achieved, treat completion as unproven and verify
it against the actual current state:
- For every explicit requirement, numbered item, named artifact, command, test,
  gate, invariant, and deliverable, identify the authoritative evidence ...
- Treat uncertain or indirect evidence as not achieved ...
- The audit must prove completion, not merely fail to find obvious remaining work.
```

我觉得最后那句 `The audit must prove completion, not merely fail to find obvious remaining work.` 最值得看：没发现明显的剩余工作，不等于证明了已经完成，举证责任压在"完成"这一侧。逐条要求都有现状证据支撑，才能调 `update_goal(complete)`。

还有一个容易忽略的设计，blocked 的迟滞阈值：

```text
Blocked audit:
- Do not call update_goal with status "blocked" the first time a blocker appears.
- Only use status "blocked" when the same blocking condition has repeated for at
  least three consecutive goal turns ...
```

第一次遇到阻塞不准喊卡，同一个阻塞连续撞三轮才允许标 `blocked`。这是在防 LLM 自主跑的另一个老毛病：遇到一点困难就缩。而且这条规则在 `continuation.md`（prompt）和 `spec.rs`（工具 schema 的 description）里写了一模一样的两份，典型的 defense in depth，模型不管从 prompt 还是从工具定义读到，拿到的都是同一条硬约束。

到这里，两边在"谁来判定完成"上的差别就很清楚了。

Humanize 1.0 是外部对抗式审查。这里得先把"吐 `COMPLETE`"讲清楚，不然容易当成一句口号。Humanize 1.0 的 RLCR 是这么转的：干活的实现者（实现者通常是 Claude）跑完一轮，把这一轮做了什么写进 `round-N-summary.md`，然后正常退出；退出的瞬间 Stop hook 触发，另起一个独立的 Codex 进程（`codex exec`，默认 `gpt-5.5:high`），把原始 plan、Claude 这轮的 summary、commit 历史、goal-tracker 一起塞给它，让它当审查员。这个审查 prompt（`regular-review.md`）要求不少：它要 "meticulous and skeptical"、"DO NOT follow its lead"（不许跟着实现者一起放水），deferred 的任务一律按未完成处理，只有当所有 AC 都满足、没有任何遗留时，才允许在审查结果的最后一行单独输出一个词 `COMPLETE`；否则就把发现的问题写进 `round-N-review-result.md`。

这里要注意，这个 `COMPLETE` 不是写给人看的，是写给 bash 看的。Stop hook 会去 grep 审查结果的最后一行：grep 不到 `COMPLETE`，就把这份 review 当成下一轮的 prompt 喂回给实现者，循环继续；grep 到了，才放行进入第二阶段 `codex review --base <branch>`，让 Codex 真正去审代码 diff，按 `[P0-9]` 标严重度，有问题接着改，没问题才收尾。所以 `COMPLETE` 更像一个机器可读的放行信号：判"完成"的那个模型实例跟干活的不是同一个，实现者无权自己宣布完成。

`/goal` 是同模型自审。没有第二个模型、没有第二个进程、没有 `COMPLETE` 这种外部 sentinel，靠的是一段把举证责任压到"完成"侧的 prompt，让同一个模型在调 `update_goal(complete)` 之前先做一遍逐条审计。放行信号就是这次工具调用本身。

如果只看防止过早收工，我会更信 Humanize 1.0。干活的模型和审查的模型不是同一个，前者想放行也得先过后者那一关。`/goal` 的好处是省进程、省 token，但它还是同一个模型在给自己打分；prompt 写得再严，也和另起一个模型挑刺不一样。

# 0x8. 定位辨析：/goal 与 Humanize 1.0 的关系

看完代码后，我觉得不能把 `/goal` 理解成 Humanize 1.0 的内置替代品。它先解决 Codex 自己的运行时问题：goal 放哪、线程空下来怎么接着跑、token 怎么记、出错后怎么停住。

Humanize 1.0 操心的是另一段流程：一轮做完之后，谁来审，审查材料怎么存，后面怎么根据审查结果继续。它有两块 `/goal` 现在没有的东西：

1. 独立审查。`/goal` 是自审，Humanize 1.0 是请第二个模型来挑刺。只看"防止过早完成"这件事，外部审查更硬一点。
2. 外部账本。Humanize 1.0 每一轮都把状态落到文件：`goal-tracker.md` 里有不可变的目标加 AC、可变的 Active/Completed/Deferred 任务，还有 Plan Evolution Log；每轮也有 `round-N-summary.md`、`round-N-review-result.md` 和 BitLessons。人可以翻，git 可以 diff，下一轮审查也有材料可读。`/goal` 的状态里只有 objective、status、预算，没有"这一轮干了什么、审查发现了什么、为什么 defer 了某个任务"这种记录。线程上下文虽然连续，但它会被压缩、会滚出窗口，不能当几十轮任务的账本用。

所以我更愿意把两者看成可以叠在一起的两层：`/goal` 负责让活线程自己续上、预算撞线就停、错误不要反复烧 token；Humanize 1.0 保留外部审查和轮次记录。这样说比"谁替代谁"更贴近代码里的分工。

# 0x9. 值得借鉴的工程细节

不说大框架，`/goal` 代码里有几处工程细节值得借鉴：

1. 续跑靠转向注入，不重开进程。`InternalModelContextFragment` 打个 `goal` 的 source 标签插进活线程，上下文零重建。能不能保活线程，直接决定了你的 continuation prompt 是 5 行还是 50 行。

2. objective 当不可信数据处理。三个 prompt 模板都把它标成 `user-provided data, not higher-priority instructions`，包在 `<objective>` 标签里，渲染前 `escape_xml_text` 转义 `& < >`。一个会被持久化、跨很多轮反复喂给模型的字符串，本身就是个长期攻击面，这个防护是必要的。

3. token 记账减掉缓存命中，`input - cached_input + output`。长任务别把缓存 input 算进预算，否则会被"反复读上下文"吃光。

4. 状态翻转做成原子 SQL。累加用量和判定 `budget_limited` 在同一条 UPDATE 里，配一把信号量串行化并发记账。事件驱动加并发回调的场景，这种 race-free 写法省掉大量诡异 bug。

5. 出错熔断到 Blocked。不可重试的错误（尤其 compaction error）直接把 goal 打成 Blocked，掐断 idle 自动续跑的死循环。自主长跑最怕的就是"在同一个错误上无限烧钱"。

6. 同一条硬约束写两份。blocked 的"连续三轮"阈值在 prompt 和工具 schema 里各写一遍，模型从哪个口子读到都是同一条规则。

# 0xA. 参考

- Codex 源码（本文 commit `04483f4`）：https://github.com/openai/codex ，goal 实现集中在 `codex-rs/ext/goal/`、`codex-rs/state/src/.../thread_goal.rs`、`codex-rs/prompts/templates/goals/`
- Codex Goals cookbook（OpenAI Cookbook，`using_goals_in_codex.ipynb`）：https://github.com/openai/openai-cookbook
- Humanize 1.0（RLCR）：https://github.com/PolyArch/humanize
