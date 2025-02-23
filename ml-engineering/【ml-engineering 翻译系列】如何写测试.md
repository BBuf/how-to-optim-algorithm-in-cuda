> 我的课程笔记，欢迎关注：https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/cuda-mode 
> 本篇文档的来源：https://github.com/stas00/ml-engineering 。

# 编写和运行测试

注意：本文档的一部分内容涉及到包含的testing_utils.py(https://github.com/stas00/ml-engineering/blob/master/testing/testing_utils.py)提供的功能,这些功能大部分是我在HuggingFace工作期间开发的。

本文档涵盖了`pytest`和`unittest`的功能,并展示了如何将两者结合使用。


## 运行测试

### 运行所有测试

```console
pytest
```
我使用以下别名:
```bash
alias pyt="pytest --disable-warnings --instafail -rA"
```

这告诉 pytest:

- 禁用警告
- `--instafail` 在失败发生时立即显示,而不是在最后显示
- `-rA` 生成简短的测试摘要信息

它需要你安装:
```
pip install pytest-instafail
```


### 获取所有测试列表

显示测试套件中的所有测试:

```bash
pytest --collect-only -q
```

显示给定测试文件中的所有测试:

```bash
pytest tests/test_optimization.py --collect-only -q
```

我使用以下别名:
```bash
alias pytc="pytest --disable-warnings --collect-only -q"
```

### 运行单个测试模块

运行单个测试模块:

```bash
pytest tests/utils/test_logging.py
```

### 运行特定测试

如果使用 `unittest`,要运行特定的子测试,你需要知道包含这些测试的 `unittest` 类的名称。例如,可以是:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

这里:

- `tests/test_optimization.py` - 包含测试的文件
- `OptimizationTest` - 测试类的名称
- `test_adam_w` - 具体测试函数的名称

如果文件包含多个类,你可以选择只运行指定类中的测试。例如:

```bash
pytest tests/test_optimization.py::OptimizationTest
```

将运行该类中的所有测试。

如前所述,你可以通过运行以下命令查看 `OptimizationTest` 类中包含的所有测试:

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

你可以通过关键字表达式运行测试。

运行仅包含 `adam` 的测试:

```bash
pytest -k adam tests/test_optimization.py
```

逻辑 `and` 和 `or` 可以用来指示是否所有关键字都匹配或仅一个匹配。`not` 可以用来否定。

运行所有不包含 `adam` 的测试:

```bash
pytest -k "not adam" tests/test_optimization.py
```

你还可以将两个模式组合在一起:

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如,要运行 `test_adafactor` 和 `test_adam_w` 你可以使用:

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

注意,我们在这里使用 `or` ,因为我们希望任一关键字匹配以包含两者。

如果你希望仅包含包含两个模式的测试,则使用 `and`:

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 仅运行修改过的测试

你可以使用 pytest-picked(https://github.com/anapaulagomes/pytest-picked) 来运行与未暂存文件或当前分支(根据 Git)相关的测试。这是一种快速测试你的更改是否破坏了任何内容的好方法,因为它不会运行与你未触及的文件相关的测试。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

所有测试将从修改但尚未提交的文件和文件夹中运行。

### 自动重新运行失败的测试


pytest-xdist(https://github.com/pytest-dev/pytest-xdist) 提供了一个非常有用的功能,可以检测所有失败的测试,然后等待你修改文件并在你修复它们时持续重新运行这些失败的测试,直到它们通过。这样你就不需要在修复后重新启动 pytest。这个过程会重复进行,直到所有测试都通过,之后会再次执行完整的运行。

```bash
pip install pytest-xdist
```

要进入模式: `pytest -f` 或 `pytest --looponfail`

文件更改通过查看 `looponfailroots` 根目录及其所有内容(递归)来检测。
如果默认值对此不起作用,你可以在项目中通过在 `setup.cfg` 中设置配置选项来更改它:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

或 `pytest.ini`/`tox.ini` 文件:

```ini
[pytest]
looponfailroots = transformers tests
```

这将导致仅在相对于 ini 文件目录的相应目录中查找文件更改。

pytest-watch(https://github.com/joeyespo/pytest-watch) 是此功能的另一种实现。


### 跳过测试模块

如果你想运行所有测试模块,除了几个你可以在运行时通过给出一个测试列表来排除它们。例如,要运行所有不包含 `test_modeling_*.py` 的测试:

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

### 清除状态

CI 构建和当隔离很重要(速度)时,缓存应该被清除:

```bash
pytest --cache-clear tests
```

### 并行运行测试

如前所述,`make test` 通过 `pytest-xdist` 插件并行运行测试(`-n X` 参数,例如 `-n 2` 运行2个并行作业)。

`pytest-xdist` 的 `--dist=` 选项允许控制测试如何分组。`--dist=loadfile` 将位于同一文件中的测试放在同一进程中。

由于执行测试的顺序不同且不可预测,如果使用 `pytest-xdist` 运行测试套件产生失败(意味着我们有一些未检测到的耦合测试),请使用 pytest-replay(https://github.com/ESSS/pytest-replay) 以相同的顺序重放测试,这应该有助于将失败序列减少到最小。

### 测试顺序和重复

按顺序、随机或分组多次重复测试是很好的做法,可以检测任何潜在的相互依赖和状态相关的错误(tear down)。而简单的多次重复也有助于发现一些被深度学习随机性掩盖的问题。

#### 重复测试

- pytest-flakefinder(https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

然后多次运行每个测试(默认50次):

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

注意: 此插件不适用于 `pytest-xdist` 的 `-n` 标志。

注意: 还有另一个插件 `pytest-repeat`,但它不适用于 `unittest`。


#### 随机运行测试

```bash
pip install pytest-random-order
```

重要提示:安装 `pytest-random-order` 后会自动随机化测试,无需任何配置更改或命令行选项。

如前所述,这允许检测耦合的测试 - 即一个测试的状态会影响另一个测试的状态。当安装了 `pytest-random-order` 后,它会打印该会话使用的随机种子,例如:

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

这样,如果特定的测试序列失败了,你可以通过添加相同的种子来重现它,例如:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```
只有在使用完全相同的测试列表(或根本没有列表)时,它才会重现完全相同的顺序。一旦你开始手动缩小列表范围,就不能再依赖种子了,而是必须按照它们失败的确切顺序手动列出它们,并使用 `--random-order-bucket=none` 告诉 pytest 不要随机化它们,例如:

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

要禁用所有测试的随机化:

```bash
pytest --random-order-bucket=none
```

默认情况下,隐含 `--random-order-bucket=module` ,这将按模块级别打乱文件。它还可以在 `class`、`package`、`global` 和 `none` 级别打乱。有关详细信息,请参阅其文档(https://github.com/jbasko/pytest-random-order)。

另一个随机化替代方案是: `pytest-randomly`(https://github.com/pytest-dev/pytest-randomly)。此模块具有非常相似的功能/接口,但没有 `pytest-random-order` 中的桶模式。它也有在安装后强制自己的问题。

### 外观变化

#### pytest-sugar

pytest-sugar(https://github.com/Frozenball/pytest-sugar) 是一个插件,可以改善外观,添加进度条,并立即显示失败的测试和断言。它在安装后自动激活。

```bash
pip install pytest-sugar
```

要运行没有它的测试,运行:

```bash
pytest -p no:sugar
```

或卸载它。



#### 报告每个子测试名称及其进度

对于单个或一组测试通过 `pytest` (在 `pip install pytest-pspec` 之后):

```bash
pytest --pspec tests/test_optimization.py
```

#### 立即显示失败的测试

pytest-instafail(https://github.com/pytest-dev/pytest-instafail) 显示失败和错误,而不是等待测试会话结束。

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### 是否使用GPU

在 GPU 设置上,要测试 CPU 模式,添加 `CUDA_VISIBLE_DEVICES=""`:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

或者如果你有多个 GPU,你可以通过 `pytest` 指定使用哪一个。例如,要仅使用第二个 GPU(如果你有 GPU `0` 和 `1`),你可以运行:

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

这在你想在不同的 GPU 上运行不同任务时非常有用。

有些测试必须在 CPU 上运行,有些在 CPU 或 GPU 或 TPU 上运行,有些在多个 GPU 上运行。以下跳过装饰器用于设置测试的 CPU/GPU/TPU 要求:

- `require_torch` - 此测试仅在 torch 下运行
- `require_torch_gpu` - 与 `require_torch` 相同,但需要至少 1 个 GPU
- `require_torch_multi_gpu` - 与 `require_torch` 相同,但需要至少 2 个 GPU
- `require_torch_non_multi_gpu` - as `require_torch` plus requires 0 or 1 GPUs
- `require_torch_up_to_2_gpus` - 与 `require_torch` 相同,但需要 0 或 1 或 2 个 GPU
- `require_torch_tpu` - 与 `require_torch` 相同,但需要至少 1 个 TPU

让我们在以下表格中描绘 GPU 要求:


| n gpus | decorator                      |
|--------|--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


例如,这里是一个仅在有 2 个或更多 GPU 且安装了 pytorch 时运行的测试:

```python no-style
from testing_utils import require_torch_multi_gpu

@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

这些装饰器可以堆叠:

```python no-style
from testing_utils import require_torch_gpu

@require_torch_gpu
@some_other_decorator
def test_example_slow_on_gpu():
```

一些装饰器如 `@parametrized` 会重写测试名称,因此 `@require_*` 跳过装饰器必须最后列出,以使其正常工作。这里是一个正确使用的示例:

```python no-style
from testing_utils import require_torch_multi_gpu
from parameterized import parameterized

@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

此顺序问题不存在于 `@pytest.mark.parametrize` 中,你可以将其放在第一个或最后一个,它仍然有效。但它仅适用于非 `unittest` 测试。

在测试中:

- 有多少个 GPU:

```python
from testing_utils import get_gpu_count

n_gpu = get_gpu_count()
```


### 分布式训练

`pytest` 不能直接处理分布式训练。如果尝试这样做 - 子进程没有做正确的事情,最终认为它们是 `pytest` 并开始在循环中运行测试套件。然而,如果一个正常进程被启动,然后启动多个工作进程并管理 IO 管道,它就会工作。

以下是一些使用它的测试:

- test_trainer_distributed.py(https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/trainer/test_trainer_distributed.py)
- test_deepspeed.py(https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/deepspeed/test_deepspeed.py)

要直接跳到执行点,在那些测试中搜索 `execute_subprocess_async` 调用,你会在 `testing_utils.py`(https://github.com/stas00/ml-engineering/blob/master/testing/testing_utils.py) 中找到它。

你需要至少 2 个 GPU 才能看到这些测试:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

(`RUN_SLOW` 是一个由 HF Transformers 使用的特殊装饰器,用于正常跳过重量级测试)

### 输出捕获

在测试执行期间,发送到 `stdout` 和 `stderr` 的任何输出都会被捕获。如果测试或设置方法失败,其相应的捕获输出通常会与失败的回溯信息一起显示。

要禁用输出捕获并正常获取 `stdout` 和 `stderr`,请使用 `-s` 或 `--capture=no`:

```bash
pytest -s tests/utils/test_logging.py
```

要将测试结果输出为 JUnit 格式:

```bash
py.test tests --junitxml=result.xml
```

### 颜色控制

要没有颜色(例如,黄色在白色背景上不可读):

```bash
pytest --color=no tests/utils/test_logging.py
```

### 发送测试报告到在线粘贴服务

为每个测试失败创建一个 URL:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

这将把测试运行信息提交到远程粘贴服务,并为每个失败提供一个 URL。你可以像往常一样选择测试,或者添加 -x 参数如果你只想发送一个特定的失败。

为整个测试会话日志创建一个 URL:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## 编写测试

大多数情况下,在同一个测试套件中组合使用 `pytest` 和 `unittest` 是可以正常工作的。你可以在这里(https://docs.pytest.org/en/stable/unittest.html)阅读支持哪些功能,但需要记住的重要一点是大多数 `pytest` fixtures都不起作用。参数化也不行,但我们使用 `parameterized` 模块,它的工作方式类似。

### 参数化

通常,需要多次运行相同的测试,但使用不同的参数。这可以在测试内部完成,但这样就无法只针对一组参数运行该测试。

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        assert_equal(math.floor(input), expected)
```

默认情况下,这个测试将运行3次,每次 `test_floor` 的最后3个参数会被赋予参数列表中对应的参数值。

你可以只运行 `negative` 和 `integer` 这两组参数的测试,使用如下命令:

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

或者运行除了 `negative` 之外的所有子测试,使用如下命令:

```bash
pytest -k "not negative" tests/test_mytest.py
```

除了使用刚才提到的 `-k` 过滤器之外,你还可以找出每个子测试的确切名称,并使用它们的确切名称来运行任何一个或所有子测试。

```bash
pytest test_this1.py --collect-only -q
```

它会列出:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

现在你可以只运行2个特定的子测试:

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

parameterized 模块(https://pypi.org/project/parameterized/)可以同时用于 `unittest` 和 `pytest` 测试。

但是,如果测试不是 `unittest`,你可以使用 `pytest.mark.parametrize`。

下面是同样的例子,这次使用 `pytest` 的 `parametrize` 标记:

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert_equal(math.floor(input), expected)
```

与 `parameterized` 类似,如果 `-k` 过滤器不能满足需求,你可以使用 `pytest.mark.parametrize` 来精确控制运行哪些子测试。不过,这个参数化函数会为子测试创建一个略有不同的命名集。它们看起来是这样的:

```bash
pytest test_this2.py --collect-only -q
```

它会列出:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

现在你可以运行特定的测试:

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

就像前面的例子一样。

### 文件和目录

在测试中,我们经常需要知道相对于当前测试文件的位置,这并不简单,因为测试可能从多个目录调用,或者可能位于不同深度的子目录中。一个帮助类 `testing_utils.TestCasePlus` 通过解决所有基本路径并提供对它们的简单访问器来解决这个问题:

- `pathlib` 对象(所有完全解析):

  - `test_file_path` - 当前测试文件路径,即 `__file__`
  - `test_file_dir` - 包含当前测试文件的目录
  - `tests_dir` - `tests` 测试套件的目录
  - `examples_dir` - `examples` 测试套件的目录
  - `repo_root_dir` - 仓库的目录
  - `src_dir` - `src` 目录(即 `transformers` 子目录所在的目录)

- 字符串化的路径 -- 与上面相同,但这些返回字符串路径,而不是 `pathlib` 对象:

  - `test_file_path_str`
  - `test_file_dir_str`
  - `tests_dir_str`
  - `examples_dir_str`
  - `repo_root_dir_str`
  - `src_dir_str`

要开始使用这些,你只需要确保测试位于 `testing_utils.TestCasePlus` 的子类中。例如:

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

如果你不需要通过 `pathlib` 操作路径,或者你只需要一个字符串路径,你可以总是调用 `pathlib` 对象的 `str()` 方法,或者使用以 `_str` 结尾的访问器。例如:

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

#### 临时文件和目录

使用唯一的临时文件和目录对于并行测试运行是必不可少的,这样测试就不会覆盖彼此的数据。我们也希望在每个创建它们的测试结束时删除临时文件和目录。因此,使用像 `tempfile` 这样的包来满足这些需求是必不可少的。

然而,当你调试测试时,你需要能够看到进入临时文件或目录的内容,并知道它的确切路径,而不是在每次测试重新运行时随机化。

一个帮助类 `testing_utils.TestCasePlus` 最好用于这些目的。它是 `unittest.TestCase` 的子类,因此我们可以在测试模块中轻松继承它。

这里是一个使用它的例子:

```python
from testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

这段代码创建一个唯一的临时目录,并将 `tmp_dir` 设置为其位置。

- 创建一个唯一的临时目录:

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir` 将包含创建的临时目录的路径。它将在测试结束时自动删除。

- 创建一个我选择的临时目录,确保在测试开始前它是空的,并且在测试结束后不要清空它。

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

这在调试时很有用,当你想要监控一个特定的目录,并确保之前的测试没有在其中留下任何数据时。

- 你可以通过直接重写 `before` 和 `after` 参数来覆盖默认行为,导致以下行为之一:

  - `before=True`: 临时目录将在测试开始时总是被清空。
  - `before=False`: 如果临时目录已经存在,任何现有的文件将保持不变。
  - `after=True`: 临时目录将在测试结束时总是被删除。
  - `after=False`: 临时目录将在测试结束时保持不变。


footnote: 为了安全地运行等效于 `rm -r` 的操作,只有在使用明确的 `tmp_dir` 时才允许项目仓库检查点中的子目录,因此,请始终传递以 `./` 开头路径。

footnote: 每个测试可以注册多个临时目录,除非请求其他行为,否则它们都将自动删除。


#### 临时 sys.path 覆盖

如果你需要临时覆盖 `sys.path` 以从另一个测试导入,你可以使用 `ExtendSysPath` 上下文管理器。例如:


```python
import os
from testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 跳过测试

这在发现一个错误并编写一个新的测试时很有用,但错误尚未修复。为了能够将其提交到主仓库,我们需要确保在 `make test` 期间跳过它。

方法:

- 一个 **skip** 意味着你期望你的测试只有在某些条件满足时才通过,否则 pytest 应该完全跳过运行测试。常见的例子是跳过只在 Windows 上运行的测试,或者跳过依赖于当前不可用的外部资源(例如数据库)的测试。

- 一个 **xfail** 意味着你期望一个测试由于某种原因失败。一个常见的例子是一个尚未实现的特性或尚未修复的错误。当一个测试在预期失败的情况下通过(用 `pytest.mark.xfail` 标记)时,它将被报告为 xpass。

两者之间的一个重要区别是 `skip` 不运行测试,而 `xfail` 会运行。因此,如果导致测试失败的代码导致了一些糟糕的状态,影响了其他测试,请不要使用 `xfail`。

#### 实现

- 以下是如何无条件地跳过整个测试:

```python no-style
@unittest.skip("this bug needs to be fixed")
def test_feature_x():
```

或通过 pytest:

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

或通过 pytest:

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

以下是如何基于测试内部的检查跳过测试:

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

或整个模块:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)
```

或通过 pytest:

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- 以下是如何在某些导入缺失时跳过模块中的所有测试:

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

- 基于条件跳过测试:

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

或:

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

或跳过整个模块:

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

更多细节、示例和方法见(https://docs.pytest.org/en/latest/skipping.html)。



### 捕获输出

#### 捕获 stdout/stderr 输出

为了测试写入 `stdout` 和/或 `stderr` 的函数,测试可以访问这些流,使用 `pytest` 的 `capsys` 系统(https://docs.pytest.org/en/latest/capture.html)。以下是如何实现的:

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # consume the captured output streams
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # test:
    assert msg in out
    assert msg in err
```

当然,大多数情况下,`stderr` 会作为异常的一部分出现,因此在这种情况下必须使用 try/except:

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "Not a good value"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
        assert msg in error, f"{msg} is in the exception:\n{error}"
```

另一种捕获 stdout 的方法是通过 `contextlib.redirect_stdout`:

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # optional: if you want to replay the consumed streams:
    sys.stdout.write(out)
    # test:
    assert msg in out
```

捕获 stdout 的一个潜在问题是它可能包含 `\r` 字符,在正常的 `print` 中,这些字符会重置已经打印的所有内容。这对 `pytest` 没有问题,但对 `pytest -s` 这些字符会被包含在缓冲区中,因此为了能够使用和没有 `-s` 的测试运行,必须对捕获的输出进行额外的清理,使用 `re.sub(r'~.*\r', '', buf, 0, re.M)`。

但是,然后我们有一个帮助类上下文管理器包装器,无论是否包含某些 `\r` 字符,都可以自动处理所有内容,因此很简单:

```python
from testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

以下是一个完整的测试示例:

```python
from testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

如果你想捕获 `stderr` 使用 `CaptureStderr` 类:

```python
from testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

如果你想同时捕获两个流,使用父类 `CaptureStd` 类:

```python
from testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

另外,为了帮助调试测试问题,默认情况下这些上下文管理器在退出上下文时自动重放捕获的流。


#### 捕获 logger 流

如果你想验证 logger 的输出,可以使用 `CaptureLogger`:

```python
from transformers import logging
from testing_utils import CaptureLogger

msg = "Testing 1, 2, 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
assert cl.out, msg + "\n"
```

### 使用环境变量测试

如果你想测试特定测试的环境变量影响,可以使用一个帮助装饰器 `transformers.testing_utils.mockenv`

```python
from testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

有时需要调用外部程序,这需要设置 `PYTHONPATH` 在 `os.environ` 中包含多个本地路径。一个帮助类 `testing_utils.TestCasePlus` 来帮助:

```python
from testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # now call the external program, passing `env` to it
```

根据测试文件是否在 `tests` 测试套件或 `examples` 测试套件中,它会正确设置 `env[PYTHONPATH]` 以包含其中一个目录,并设置 `src` 目录以确保测试是针对当前仓库进行的,最后,如果测试被调用之前有任何 `env[PYTHONPATH]` 设置。

这个帮助方法创建一个 `os.environ` 对象的副本,所以原来的保持不变。


### Getting reproducible results

在某些情况下,你可能希望为测试移除随机性。为了获得相同的可重复结果,需要固定种子:

```python
seed = 42

# python RNG
import random

random.seed(seed)

# pytorch RNGs
import torch

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# numpy RNG
import numpy as np

np.random.seed(seed)

# tf RNG
tf.random.set_seed(seed)
```

## 调试测试

为了在警告点开始调试器,这样做:

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```


## 一个巨大的 hack 来创建多个 pytest 报告

以下是一个巨大的 `pytest` 补丁,我多年前做的,以帮助更好地理解 CI 报告。

为了激活它,添加到 `tests/conftest.py` (或创建它,如果你还没有):

```python
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

然后当你运行测试套件时,像这样添加 `--make-reports=mytests`:

```bash
pytest --make-reports=mytests tests
```

然后它会创建 8 个单独的报告:

```bash
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
```

所以现在不再是只有一个来自 `pytest` 的输出,而是每个类型报告保存到各自文件中。

这个功能在 CI 中最有用,使它更容易同时检查问题和查看和下载单独的报告。

使用不同的值到 `--make-reports=` 为不同的测试组可以分别保存,而不是相互覆盖。

所有这些功能已经在 `pytest` 中,但没有办法轻松提取,所以我添加了猴子补丁重写 `testing_utils.py`(https://github.com/stas00/ml-engineering/blob/master/testing/testing_utils.py)。好吧,我问如果我可以贡献这个作为 `pytest` 的功能,但我的提议不受欢迎。


## testing_utils.py 解析

```markdown
# I developed the bulk of this library while I worked at HF

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio  # noqa
import contextlib
import importlib.util
import inspect
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from distutils.util import strtobool
from io import StringIO
from pathlib import Path
from typing import Iterator, Union
from unittest import mock
from unittest.case import SkipTest

import numpy as np
from packaging import version
from parameterized import parameterized


# 尝试导入PyTorch,如果成功则设置_torch_available为True,否则为False
try:
    import torch
    _torch_available = True
except Exception:
    _torch_available = False


# 返回是否可以使用PyTorch的标志
def is_torch_available():
    return _torch_available


# 从环境变量中解析布尔标志,如果未设置则使用默认值
def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # 如果环境变量未设置,使用默认值
        _value = default
    else:
        # 如果环境变量已设置,将其转换为True或False
        try:
            _value = strtobool(value)
        except ValueError:
            # 支持更多值,但保持错误消息简单
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


# 从环境变量中解析整数值,如果未设置则使用默认值
def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            raise ValueError(f"If set, {key} must be a int.")
    return _value


# 装饰器:标记需要PyTorch的测试
def require_torch(test_case):
    """
    装饰器标记需要PyTorch的测试。
    当PyTorch未安装时,这些测试会被跳过。
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)
    else:
        return test_case


# 装饰器:标记需要无GPU环境的测试
def require_torch_no_gpus(test_case):
    """
    装饰器标记需要无GPU设置的测试(在PyTorch中)。这些测试在有GPU的机器上会被跳过。
    要仅运行无gpu测试,假设所有测试名称包含no_gpu: $ pytest -sv ./tests -k "no_gpu"
    """
    import torch

    if is_torch_available() and torch.cuda.device_count() > 0:
        return unittest.skip("test requires an environment w/o GPUs")(test_case)
    else:
        return test_case


# 装饰器:标记需要多GPU环境的测试
def require_torch_multi_gpu(test_case):
    """
    装饰器标记需要多GPU设置的测试(在PyTorch中)。这些测试在没有多个GPU的机器上会被跳过。
    要仅运行多gpu测试,假设所有测试名称包含multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() < 2:
        return unittest.skip("test requires multiple GPUs")(test_case)
    else:
        return test_case


# 装饰器:标记需要0或1个GPU环境的测试
def require_torch_non_multi_gpu(test_case):
    """
    装饰器标记需要0或1个GPU设置的测试(在PyTorch中)。
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 1:
        return unittest.skip("test requires 0 or 1 GPU")(test_case)
    else:
        return test_case


# 装饰器:标记需要0-2个GPU环境的测试
def require_torch_up_to_2_gpus(test_case):
    """
    装饰器标记需要0或1或2个GPU设置的测试(在PyTorch中)。
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 2:
        return unittest.skip("test requires 0 or 1 or 2 GPUs")(test_case)
    else:
        return test_case


# 如果PyTorch可用,设置设备为cuda或cpu
if is_torch_available():
    # 设置环境变量CUDA_VISIBLE_DEVICES=""强制使用cpu模式
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch_device = None


# 装饰器:标记需要CUDA和PyTorch的测试
def require_torch_gpu(test_case):
    """装饰器标记需要CUDA和PyTorch的测试。"""
    if torch_device != "cuda":
        return unittest.skip("test requires CUDA")(test_case)
    else:
        return test_case


# 检查是否可以使用deepspeed
def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


# 装饰器:标记需要deepspeed的测试
def require_deepspeed(test_case):
    """
    装饰器标记需要deepspeed的测试
    """
    if not is_deepspeed_available():
        return unittest.skip("test requires deepspeed")(test_case)
    else:
        return test_case


# 检查是否可以使用bitsandbytes
def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


# 装饰器:标记需要bitsandbytes的测试
def require_bnb(test_case):
    """
    装饰器标记需要bitsandbytes的测试
    """
    if not is_bnb_available():
        return unittest.skip("test requires bitsandbytes from https://github.com/facebookresearch/bitsandbytes")(
            test_case
        )
    else:
        return test_case


# 非装饰器函数:如果缺少bitsandbytes则跳过测试
def require_bnb_non_decorator():
    """
    非装饰器函数,如果缺少bitsandbytes则跳过测试
    """
    if not is_bnb_available():
        raise SkipTest("Test requires bitsandbytes from https://github.com/facebookresearch/bitsandbytes")


# 设置随机种子以实现可重现性
def set_seed(seed: int = 42):
    """
    帮助函数,用于设置random、numpy、torch的种子以实现可重现的行为

    Args:
        seed (:obj:`int`): 要设置的种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ 即使cuda不可用也可以安全调用此函数


# 获取可用的GPU数量
def get_gpu_count():
    """
    返回可用的gpu数量(无论使用torch还是tf)
    """
    if is_torch_available():
        import torch
        return torch.cuda.device_count()
    else:
        return 0


# 比较两个张量或非张量数字是否相等
def torch_assert_equal(actual, expected, **kwargs):
    """
    比较两个张量或非张量数字是否相等
    """
    # assert_close在pt-1.9左右添加,它进行更好的检查 - 例如会检查维度是否匹配
    return torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, **kwargs)


# 比较两个张量或非张量数字是否接近
def torch_assert_close(actual, expected, **kwargs):
    """
    比较两个张量或非张量数字是否接近。
    """
    # assert_close在pt-1.9左右添加,它进行更好的检查 - 例如会检查维度是否匹配
    return torch.testing.assert_close(actual, expected, **kwargs)


# 检查是否支持torch bf16
def is_torch_bf16_available():
    # 从 https://github.com/huggingface/transformers/blob/26eb566e43148c80d0ea098c76c3d128c0281c16/src/transformers/file_utils.py#L301
    if is_torch_available():
        import torch

        if not torch.cuda.is_available() or torch.version.cuda is None:
            return False
        if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
            return False
        if int(torch.version.cuda.split(".")[0]) < 11:
            return False
        if not version.parse(torch.__version__) >= version.parse("1.09"):
            return False
        return True
    else:
        return False


# 装饰器:标记需要支持bf16的CUDA硬件和PyTorch >= 1.9的测试
def require_torch_bf16(test_case):
    """装饰器标记需要支持bf16的CUDA硬件和PyTorch >= 1.9的测试。"""
    if not is_torch_bf16_available():
        return unittest.skip("test requires CUDA hardware supporting bf16 and PyTorch >= 1.9")(test_case)
    else:
        return test_case


def get_tests_dir(append_path=None):
    """
    获取测试目录的完整路径。

    Args:
        append_path: 可选的路径,会被追加到tests目录路径后面

    Return:
        返回tests目录的完整路径,这样测试可以从任何位置调用。
        如果提供了append_path参数,它会被追加到tests目录后面。

    注意:
        - 使用inspect.stack()获取调用者的文件路径
        - 使用os.path.abspath获取绝对路径
        - 使用os.path.dirname获取目录名
        - 如果提供了append_path,使用os.path.join拼接路径
    """
    # 获取调用此函数的文件路径
    caller__file__ = inspect.stack()[1][1]
    # 获取tests目录的绝对路径
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))
    if append_path:
        # 如果提供了append_path,将其追加到tests目录后
        return os.path.join(tests_dir, append_path)
    else:
        # 否则直接返回tests目录路径
        return tests_dir


def parameterized_custom_name_func_join_params(func, param_num, param):
    """
    自定义测试名称生成器函数,使所有参数都显示在子测试名称中。
    默认情况下它只显示第一个参数,或者对于多个参数只使用唯一的ID序列而不显示任何参数。

    用法:

    @parameterized.expand(
        [
            (0, True),
            (0, False), 
            (1, True),
        ],
        name_func=parameterized_custom_name_func_join_params,
    )
    def test_determinism_wrt_rank(self, num_workers, pad_dataset):

    将生成:

    test_determinism_wrt_rank_0_true
    test_determinism_wrt_rank_0_false 
    test_determinism_wrt_rank_1_true

    """
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"



class CaptureStd:
    """
    用于捕获标准输出和标准错误的上下文管理器。

    可以捕获:
    - stdout: 重放输出,清理并通过 obj.out 获取
    - stderr: 重放输出并通过 obj.err 获取 
    - combined: 合并选定的流并通过 obj.combined 获取

    初始化参数:
    - out - 是否捕获stdout: True/False,默认 True
    - err - 是否捕获stderr: True/False,默认 True 
    - replay - 是否重放: True/False,默认 True。默认情况下每个捕获的流在上下文退出时会重放,
      这样可以看到测试在做什么。如果不需要这个行为,可以传入 replay=False 禁用此功能。
    """

    def __init__(self, out=True, err=True, replay=True):
        # 是否重放捕获的输出
        self.replay = replay

        # 初始化stdout捕获
        if out:
            self.out_buf = StringIO()  # 创建stdout缓冲区
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        # 初始化stderr捕获  
        if err:
            self.err_buf = StringIO()  # 创建stderr缓冲区
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

            self.combined = "error: CaptureStd context is unfinished yet, called too early"

    def __enter__(self):
        # 重定向stdout
        if self.out_buf is not None:
            self.out_old = sys.stdout  # 保存原始stdout
            sys.stdout = self.out_buf  # 重定向到缓冲区

        # 重定向stderr
        if self.err_buf is not None:
            self.err_old = sys.stderr  # 保存原始stderr
            sys.stderr = self.err_buf  # 重定向到缓冲区

        self.combined = ""  # 初始化合并输出

        return self

    def __exit__(self, *exc):
        # 恢复stdout并获取捕获的输出
        if self.out_buf is not None:
            sys.stdout = self.out_old  # 恢复原始stdout
            captured = self.out_buf.getvalue()  # 获取捕获的输出
            if self.replay:
                sys.stdout.write(captured)  # 重放输出
            self.out = apply_print_resets(captured)  # 清理输出
            self.combined += self.out  # 添加到合并输出

        # 恢复stderr并获取捕获的输出
        if self.err_buf is not None:
            sys.stderr = self.err_old  # 恢复原始stderr
            captured = self.err_buf.getvalue()  # 获取捕获的输出
            if self.replay:
                sys.stderr.write(captured)  # 重放输出
            self.err = captured  # 保存输出
            self.combined += self.err  # 添加到合并输出

    def __repr__(self):
        # 生成捕获内容的字符串表示
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


# 在测试中最好只捕获需要的流,否则容易遗漏内容。
# 除非需要同时捕获两个流,否则使用下面的子类(代码更简洁)。
# 或者配置 CaptureStd 禁用不需要测试的流。


class CaptureStdout(CaptureStd):
    """与CaptureStd相同但只捕获stdout"""

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):
    """与CaptureStd相同但只捕获stderr"""

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:
    """
    用于捕获`logging`流

    这个类用于在测试中捕获logging模块的日志输出。它通过添加一个临时的StreamHandler来捕获日志,
    并在上下文管理器退出时移除该handler。

    Args:
        - logger: `logging` logger对象,要捕获的logger实例

    Results:
        捕获的输出可以通过 `self.out` 获取

    Example::

        >>> from transformers import logging
        >>> from transformers.testing_utils import CaptureLogger

        >>> msg = "Testing 1, 2, 3"
        >>> logging.set_verbosity_info()
        >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart") 
        >>> with CaptureLogger(logger) as cl:
        ...     logger.info(msg)
        >>> assert cl.out, msg+"\n"
    """

    def __init__(self, logger):
        # 保存logger实例
        self.logger = logger
        # 创建StringIO对象用于存储捕获的输出
        self.io = StringIO()
        # 创建StreamHandler将输出重定向到StringIO
        self.sh = logging.StreamHandler(self.io)
        # 存储捕获的输出
        self.out = ""

    def __enter__(self):
        # 进入上下文时添加handler开始捕获
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        # 退出上下文时移除handler
        self.logger.removeHandler(self.sh)
        # 获取捕获的输出
        self.out = self.io.getvalue()

    def __repr__(self):
        # 返回捕获内容的字符串表示
        return f"captured: {self.out}\n"


@contextlib.contextmanager
# adapted from https://stackoverflow.com/a/64789046/9201239
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    临时将给定路径添加到`sys.path`。

    这个上下文管理器用于临时修改Python的模块搜索路径。它将提供的路径添加到sys.path的开头,
    这样Python就可以从这个新添加的路径中导入模块。当上下文退出时,会自动将该路径从sys.path中移除。

    参数:
        path: 要添加到sys.path的路径,可以是字符串或os.PathLike对象

    返回:
        Iterator[None]: 一个迭代器,用于上下文管理器的实现

    用法示例::

       with ExtendSysPath('/path/to/dir'):
           mymodule = importlib.import_module('mymodule')

    """

    # 将路径转换为字符串格式
    path = os.fspath(path)
    try:
        # 将路径插入到sys.path的开头
        sys.path.insert(0, path)
        yield
    finally:
        # 退出上下文时移除添加的路径
        sys.path.remove(path)


class TestCasePlus(unittest.TestCase):
    """
    这个类扩展了 `unittest.TestCase` 并添加了额外的功能。

    功能1: 一组完全解析的重要文件和目录路径访问器。

    在测试中我们经常需要知道相对于当前测试文件的各种路径位置,这并不简单,因为测试可能从多个目录调用或位于不同深度的子目录中。
    这个类通过整理所有基本路径并提供简单的访问器来解决这个问题:

    * `pathlib` 对象(全部完全解析):
       - `test_file_path` - 当前测试文件路径(=`__file__`)
       - `test_file_dir` - 包含当前测试文件的目录
       - `tests_dir` - `tests` 测试套件的目录
       - `data_dir` - `tests/data` 测试套件的目录
       - `repo_root_dir` - 仓库的目录
       - `src_dir` - `m4` 子目录所在的目录(在本例中与 repo_root_dir 相同)

    * 字符串化路径 - 与上面相同但返回字符串形式的路径而不是 `pathlib` 对象:
       - `test_file_path_str`
       - `test_file_dir_str` 
       - `tests_dir_str`
       - `data_dir_str`
       - `repo_root_dir_str`
       - `src_dir_str`

    功能2: 灵活的自动删除临时目录,保证在测试结束时被删除。

    1. 创建唯一的临时目录:
    ::
        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir()

    `tmp_dir` 将包含创建的临时目录的 pathlib 路径。它将在测试结束时自动删除。

    2. 创建自定义的临时目录,确保测试开始前为空,测试后不清空:
    ::
        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir("./xxx")

    这对调试很有用,当你想监控特定目录并确保之前的测试没有在其中留下任何数据时。

    3. 你可以通过直接覆盖 `before` 和 `after` 参数来覆盖前两个选项,导致以下行为:

    `before=True`: 临时目录在测试开始时总是被清空。
    `before=False`: 如果临时目录已存在,任何现有文件都将保留。
    `after=True`: 临时目录在测试结束时总是被删除。
    `after=False`: 临时目录在测试结束时总是保持不变。

    如果你想要返回非 pathlib 版本,请使用 `self.get_auto_remove_tmp_dir_str()`。

    注意1: 为了安全地运行 `rm -r` 的等效操作,如果使用显式的 `tmp_dir`,只允许项目仓库检出的子目录,
    这样就不会错误地删除 `/tmp` 或文件系统的其他重要部分。即请始终传递以 `./` 开头的路径。

    注意2: 每个测试可以注册多个临时目录,除非另有要求,否则它们都将被自动删除。

    功能3: 获取 `os.environ` 对象的副本,该副本设置了特定于当前测试套件的 `PYTHONPATH`。
    这对于从测试套件调用外部程序很有用 - 例如分布式训练。

    ::
        def test_whatever(self):
            env = self.get_env()
    """

    def setUp(self):
        """
        setUp 方法用于初始化测试类的路径相关属性。主要完成以下工作:

        1. 初始化临时目录列表:
           - self.teardown_tmp_dirs = [] 用于存储需要在测试结束时清理的临时目录

        2. 解析测试文件路径:
           - 使用 inspect.getfile() 获取当前测试类的文件路径
           - 使用 Path().resolve() 将路径转换为绝对路径
           - self._test_file_dir 存储测试文件所在目录

        3. 查找仓库根目录:
           - 向上遍历最多3层父目录
           - 检查每层目录是否同时包含 "m4" 和 "tests" 子目录
           - 找到后将该目录设为 self._repo_root_dir
           - 如果找不到则抛出 ValueError

        4. 设置其他相关目录:
           - self._tests_dir: 测试目录 (<repo_root>/tests)
           - self._data_dir: 测试数据目录 (<repo_root>/tests/test_data) 
           - self._src_dir: 源码目录,这里等同于仓库根目录
        """
        # get_auto_remove_tmp_dir 功能:
        self.teardown_tmp_dirs = []

        # 找出 repo_root、tests 等的解析路径
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self._test_file_dir = path.parents[0]
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            if (tmp_dir / "m4").is_dir() and (tmp_dir / "tests").is_dir():
                break
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        self._tests_dir = self._repo_root_dir / "tests"
        self._data_dir = self._repo_root_dir / "tests" / "test_data"
        self._src_dir = self._repo_root_dir  # m4 不在仓库中使用 "src/" 前缀

    @property
    def test_file_path(self):
        return self._test_file_path

    @property
    def test_file_path_str(self):
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        return self._tests_dir

    @property
    def tests_dir_str(self):
        return str(self._tests_dir)

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def data_dir_str(self):
        return str(self._data_dir)

    @property
    def repo_root_dir(self):
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        return self._src_dir

    @property
    def src_dir_str(self):
        return str(self._src_dir)

    def get_env(self):
        """
        返回 `os.environ` 对象的副本,该副本正确设置了 `PYTHONPATH`。这对于从测试套件调用外部程序很有用 - 例如分布式训练。

        它总是首先插入 `.`,然后根据测试套件类型插入 `./tests`,最后是预设的 `PYTHONPATH`(如果有的话)(所有完整解析的路径)。
        """
        env = os.environ.copy()
        paths = [self.src_dir_str]
        paths.append(self.tests_dir_str)
        paths.append(env.get("PYTHONPATH", ""))

        env["PYTHONPATH"] = ":".join(paths)
        return env

    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
        """
        参数:
            tmp_dir (`string`, 可选):
                如果为 `None`:
                   - 将创建唯一的临时路径
                   - 如果 `before` 为 `None` 则设置 `before=True`
                   - 如果 `after` 为 `None` 则设置 `after=True`
                否则:
                   - 将创建 `tmp_dir`
                   - 如果 `before` 为 `None` 则设置 `before=True`
                   - 如果 `after` 为 `None` 则设置 `after=False`
            before (`bool`, 可选):
                如果为 `True` 且 `tmp_dir` 已存在,确保立即清空它
                如果为 `False` 且 `tmp_dir` 已存在,任何现有文件都将保留
            after (`bool`, 可选):
                如果为 `True`,在测试结束时删除 `tmp_dir`
                如果为 `False`,在测试结束时保持 `tmp_dir` 及其内容不变

        返回:
            tmp_dir(`string`): 与通过 `tmp_dir` 传递的值相同或自动选择的临时目录的路径
        """
        if tmp_dir is not None:
            # 定义提供自定义路径时最可能需要的行为。
            # 这很可能表示调试模式,我们希望一个容易定位的目录:
            # 1. 在测试之前清空(如果已存在)
            # 2. 在测试后保持不变
            if before is None:
                before = True
            if after is None:
                after = False

            # 为避免删除文件系统的部分,只允许相对路径
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir` 只能是相对路径,即 `./some/path`,但收到 `{tmp_dir}`"
                )

            # 使用提供的路径
            tmp_dir = Path(tmp_dir).resolve()

            # 确保目录在开始时为空
            if before is True and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            tmp_dir.mkdir(parents=True, exist_ok=True)

        else:
            # 定义自动生成唯一临时路径时最可能需要的行为
            # (非调试模式),这里我们需要一个唯一的临时目录:
            # 1. 在测试之前为空(在这种情况下无论如何都会为空)
            # 2. 在测试后完全删除
            if before is None:
                before = True
            if after is None:
                after = True

            # 使用唯一的临时目录(始终为空,不考虑 `before`)
            tmp_dir = Path(tempfile.mkdtemp())

        if after is True:
            # 注册以便删除
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir

    def get_auto_remove_tmp_dir_str(self, *args, **kwargs):
        return str(self.get_auto_remove_tmp_dir(*args, **kwargs))

    def tearDown(self):
        # get_auto_remove_tmp_dir 功能:删除注册的临时目录
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []


def mockenv(**kwargs):
    """
    这是一个便捷的装饰器包装器,允许以下用法:

    @mockenv(RUN_SLOW=True, USE_TF=False) 
    def test_something():
        run_slow = os.getenv("RUN_SLOW", False)
        use_tf = os.getenv("USE_TF", False)

    另外可以参考 `mockenv_context` 来使用上下文管理器

    Args:
        **kwargs: 要临时设置的环境变量键值对

    Returns:
        返回一个 mock.patch.dict 装饰器,用于临时修改 os.environ
    """
    # 使用 mock.patch.dict 临时修改环境变量
    # os.environ - 要修改的字典
    # kwargs - 要设置的新键值对
    return mock.patch.dict(os.environ, kwargs)


# 从 https://stackoverflow.com/a/34333710/9201239 获取的代码
@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    临时就地更新 ``os.environ`` 字典。类似于 mockenv

    ``os.environ`` 字典会就地更新,以确保在所有情况下修改都能生效。

    Args:
      remove: 要删除的环境变量。
      update: 要添加/更新的环境变量和值的字典。

    Example:

    with mockenv_context(FOO="1"):
        execute_subprocess_async(cmd, env=self.get_env())
    """
    # 获取环境变量字典的引用
    env = os.environ
    # 如果没有提供更新字典,则使用空字典
    update = update or {}
    # 如果没有提供要删除的变量,则使用空列表
    remove = remove or []

    # 获取将被更新或删除的环境变量列表(update和remove的并集与当前环境变量的交集)
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # 保存需要在退出时恢复的环境变量和值
    update_after = {k: env[k] for k in stomped}
    # 获取需要在退出时删除的环境变量(update中新增但原环境中不存在的变量)
    remove_after = frozenset(k for k in update if k not in env)

    try:
        # 更新环境变量
        env.update(update)
        # 删除指定的环境变量
        [env.pop(k, None) for k in remove]
        yield
    finally:
        # 恢复之前保存的环境变量
        env.update(update_after)
        # 删除新增的环境变量
        [env.pop(k) for k in remove_after]


# --- 测试网络辅助函数 --- #


def get_xdist_worker_id():
    """
    当在 pytest-xdist 下运行时返回 worker id (整数),否则返回 0
    
    pytest-xdist 是一个 pytest 插件,用于并行运行测试。每个并行进程都有一个唯一的 worker id。
    这个函数从环境变量中获取 worker id,如果不是在 pytest-xdist 下运行则返回 0。
    """
    worker_id_string = os.environ.get("PYTEST_XDIST_WORKER", "gw0")  # 从环境变量获取 worker id,默认为 "gw0"
    return int(worker_id_string[2:])  # 去掉 "gw" 前缀,转换为整数


DEFAULT_MASTER_PORT = 10999  # 默认主端口号


def get_unique_port_number():
    """
    当测试套件在 pytest-xdist 下运行时,我们需要确保并发测试不会使用相同的端口号。
    我们可以通过使用相同的基础端口号,然后加上 xdist worker id 来实现这一点。
    如果不是在 pytest-xdist 下运行,则加 0。
    
    返回:
        int: 唯一的端口号,等于基础端口号加上 worker id
    """
    return DEFAULT_MASTER_PORT + get_xdist_worker_id()  # 基础端口号加上 worker id 得到唯一端口号


# --- test IO helper functions --- #


def write_file(file, content):
    """
    将内容写入文件
    
    Args:
        file: 要写入的文件路径
        content: 要写入的内容
    """
    with open(file, "w") as f:
        f.write(content)


def read_json_file(file):
    """
    读取并解析JSON文件
    
    Args:
        file: JSON文件路径
        
    Returns:
        解析后的JSON对象
    """
    with open(file, "r") as fh:
        return json.load(fh)


def replace_str_in_file(file, text_to_search, replacement_text):
    """
    在文件中替换指定文本
    
    Args:
        file: 要处理的文件路径
        text_to_search: 要查找的文本
        replacement_text: 替换的文本
    """
    file = Path(file)
    text = file.read_text()
    text = text.replace(text_to_search, replacement_text)
    file.write_text(text)


#-- pytest conf functions --#

"""
这是一个让 `pytest` 输出单独报告的技巧

要激活此功能,请在 `tests/conftest.py` 中添加:

```python
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

然后运行:

```python
pytest --make-reports=mytests tests
```

然后检查 `reports/mytests/` 目录下的各个报告

```python
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
```

所以现在不再是只有一个包含所有内容的 `pytest` 输出,而是可以将每种类型的报告分别保存到各自的文件中。

"""python
# 用于避免从 tests/conftest.py 和 examples/conftest.py 多次调用 - 确保只调用一次
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    此函数需要从 `conftest.py` 通过在其中定义的 `pytest_addoption` 包装器调用。

    它允许同时加载两个 `conftest.py` 文件而不会因为添加相同的 `pytest` 选项而导致失败。
    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store", 
            default=False,
            help="生成报告文件。此选项的值用作报告名称的前缀",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    在测试套件运行结束时生成多个报告 - 每个报告都保存在当前目录的专用文件中。
    报告文件以测试套件名称作为前缀。

    此函数模拟 --duration 和 -rA pytest 参数。

    此函数需要从 `conftest.py` 通过在其中定义的 `pytest_terminal_summary` 包装器调用。

    参数:
    - tr: 从 `conftest.py` 传递的 `terminalreporter`
    - id: 像 `tests` 或 `examples` 这样的唯一标识符,将被合并到最终的报告文件名中 - 
         这是必需的,因为某些作业有多次 pytest 运行,所以我们不能让它们相互覆盖。

    注意: 此函数使用了私有的 _pytest API,虽然不太可能,但如果 pytest 进行内部更改,它可能会中断 - 
    同时它调用 terminalreporter 的默认内部方法,这些方法可能被各种 `pytest-` 插件劫持并干扰。
    """
    from _pytest.config import create_terminal_writer

    if not len(id):
        id = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    # 创建报告目录和文件
    dir = f"reports/{id}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dir}/{k}.txt"
        for k in [
            "durations",
            "errors", 
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }

    # 自定义持续时间报告
    # 注意:不需要调用 pytest --durations=XX 来获取这个单独的报告
    # 改编自 https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # 秒
            f.write("最慢的持续时间\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} 个持续时间 < {durations_min} 秒被省略")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):
        # 假设报告是 --tb=long (默认),所以我们在这里将它们截断到最后一帧
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "失败简短堆栈")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # 去掉可选的前导额外帧,只保留最后一个
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # 注意:不打印任何 rep.sections 以保持报告简短

    # 使用现成的报告函数,我们只是劫持文件句柄来记录到专用文件中
    # 改编自 https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # 注意:一些 pytest 插件可能会通过劫持默认的 `terminalreporter` 来干扰(例如 pytest-instafail 就是这样做的)

    # 使用 line/short/long 样式报告失败
    config.option.tbstyle = "auto"  # 完整的 traceback
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # 简短的 traceback
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # 每个错误一行
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # 普通警告
        tr.summary_warnings()  # 最终警告

    tr.reportchars = "wPpsxXEf"  # 模拟 -rA (用于 summary_passes() 和 short_test_summary())

    # 跳过 `passes` 报告,因为它开始花费超过 5 分钟,有时在 CircleCI 上如果花费 > 10 分钟就会超时
    # (另外,这个报告中似乎没有有用的信息,我们很少需要阅读它)
    # with open(report_files["passes"], "w") as f:
    #     tr._tw = create_terminal_writer(config, f)
    #     tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # 恢复原始设置:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


# --- 分布式测试函数 --- #


class _RunOutput:
    """
    存储子进程执行结果的类
    
    属性:
        returncode: 进程返回码
        stdout: 标准输出内容
        stderr: 标准错误内容
    """
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    """
    异步读取流中的数据并调用回调函数处理
    
    Args:
        stream: 要读取的流对象
        callback: 处理每行数据的回调函数
    """
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    """
    异步执行子进程并实时处理输出流
    
    Args:
        cmd: 要执行的命令列表
        env: 环境变量字典
        stdin: 标准输入
        timeout: 超时时间(秒)
        quiet: 是否静默模式(不打印输出)
        echo: 是否打印执行的命令
        
    Returns:
        _RunOutput 对象,包含返回码和输出内容
    """
    if echo:
        print("\nRunning: ", " ".join(cmd))

    # 创建子进程
    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # 注意:使用 `wait` 处理大量数据时可能会死锁
    # https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process.wait
    #
    # 如果开始挂起,需要切换到以下代码。问题是在完成之前看不到任何数据,如果挂起将没有调试信息。
    # out, err = await p.communicate()
    # return _RunOutput(p.returncode, out, err)

    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        """将输出同时写入sink和pipe"""
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    # XXX: timeout 参数似乎没有效果
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda line: tee(line, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda line: tee(line, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    """
    执行异步子进程的同步包装函数
    
    Args:
        cmd: 要执行的命令列表
        env: 环境变量字典
        stdin: 标准输入
        timeout: 超时时间(秒)
        quiet: 是否静默模式
        echo: 是否打印执行的命令
        
    Returns:
        _RunOutput 对象
        
    Raises:
        RuntimeError: 当进程返回非零值或没有输出时
    """
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # 检查子进程是否真正运行并产生输出
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result

```