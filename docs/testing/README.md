# 编写与运行测试 {#writing-and-running-tests}

注意：本文档有一部分涉及随附的 [testing_utils.py](testing_utils.py) 所提供的功能，其中大部分内容是我在 HuggingFace 工作时开发的。

本文同时涵盖 `pytest` 和 `unittest` 的功能，并展示两者如何一起使用。


## 运行测试 {#running-tests}

### 运行所有测试 {#run-all-tests}

```bash
pytest
```
我使用下面这个别名：
```bash
alias pyt="pytest --disable-warnings --instafail -rA"
```

它告诉 pytest：

- 禁用警告
- `--instafail` 在失败发生时立即显示失败，而不是等到最后
- `-rA` 生成简短的测试摘要信息

这需要你安装：
```bash
pip install pytest-instafail
```


### 获取所有测试的列表 {#getting-the-list-of-all-tests}

显示测试套件中的所有测试：

```bash
pytest --collect-only -q
```

显示某个指定测试文件中的所有测试：

```bash
pytest tests/test_optimization.py --collect-only -q
```

我使用下面这个别名：
```bash
alias pytc="pytest --disable-warnings --collect-only -q"
```

### 运行特定的测试模块 {#run-a-specific-test-module}

要运行单个测试模块：

```bash
pytest tests/utils/test_logging.py
```

### 运行特定的测试 {#run-specific-tests}

如果使用了 `unittest`，要运行特定的子测试，你需要知道包含这些测试的 `unittest` 类的名字。例如，可能是这样：

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

这里：

- `tests/test_optimization.py` - 包含测试的文件
- `OptimizationTest` - 测试类的名称
- `test_adam_w` - 具体测试函数的名称

如果文件包含多个类，你可以选择只运行某个给定类的测试。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest
```

将会运行该类中的所有测试。

如前所述，你可以通过运行下面的命令来查看 `OptimizationTest` 类里包含哪些测试：

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

你可以用关键词表达式来运行测试。

要只运行名字中包含 `adam` 的测试：

```bash
pytest -k adam tests/test_optimization.py
```

可以使用逻辑运算符 `and` 和 `or` 来表示是要求所有关键词都匹配，还是只需匹配其中之一。`not` 可用于取反。

要运行除名字中包含 `adam` 之外的所有测试：

```bash
pytest -k "not adam" tests/test_optimization.py
```

你还可以把这两种模式组合成一个：

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如，要同时运行 `test_adafactor` 和 `test_adam_w`，你可以使用：

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

注意这里我们用的是 `or`，因为我们希望任意一个关键词匹配即可，从而把两者都包含进来。

如果你只想包含同时匹配两个模式的测试，则应使用 `and`：

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 只运行被修改的测试 {#run-only-modified-tests}

你可以使用 [pytest-picked](https://github.com/anapaulagomes/pytest-picked)，运行与未暂存文件或当前分支（依据 Git 判断）相关的测试。这是一种快速验证你的修改没有破坏任何东西的好方法，因为它不会运行与你没有改动的文件相关的测试。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

所有来自已被修改但尚未提交的文件和文件夹的测试都会被运行。

### 源码修改后自动重新运行失败的测试 {#automatically-rerun-failed-tests-on-source-modification}

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) 提供了一个非常实用的功能：检测出所有失败的测试，然后等待你修改文件，并在你修复的过程中持续重跑这些失败的测试，直到它们通过。这样你修好之后就不需要重新启动 pytest。这一过程会反复进行，直到所有测试都通过，之后再执行一次完整运行。

```bash
pip install pytest-xdist
```

进入该模式：`pytest -f` 或 `pytest --looponfail`

文件改动的检测方式是查看 `looponfailroots` 根目录及其中的全部内容（递归地）。如果这个配置的默认值不适合你，你可以在项目中通过在 `setup.cfg` 里设置一个配置选项来更改它：

```ini
[tool:pytest]
looponfailroots = transformers tests
```

或者在 `pytest.ini`/``tox.ini`` 文件中：

```ini
[pytest]
looponfailroots = transformers tests
```

这样就会只在相应目录中查找文件改动，这些目录的路径是相对于 ini 文件所在目录指定的。

[pytest-watcher](https://github.com/olzhasar/pytest-watcher) 是这一功能的另一种实现。较老的 `pytest-watch` 经常被引用，但自 2022 年以来就没有过提交了，所以请优先使用前者或上面提到的 `--looponfail`。


### 跳过某个测试模块 {#skip-a-test-module}

如果你想运行所有测试模块，但要排除其中几个，可以给出一份要运行的测试的显式列表来实现。例如，要运行除 `test_modeling_*.py` 之外的所有测试：

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

### 清除状态 {#clearing-state}

在 CI 构建中，以及当隔离性很重要（相对于速度而言）时，应当清除缓存：

```bash
pytest --cache-clear tests
```

### 并行运行测试 {#running-tests-in-parallel}

如果你的项目定义了 `make test` 目标，它通常通过 `pytest-xdist` 插件并行运行测试（`-n X` 参数，例如用 `-n 2` 并行运行 2 个作业）。

`pytest-xdist` 的 `--dist=` 选项让你可以控制测试如何分组。`--dist=loadfile` 会把位于同一个文件中的测试放到同一个进程上。

由于执行测试的顺序不同且不可预测，如果用 `pytest-xdist` 运行测试套件时出现了失败（说明存在某些尚未被发现的耦合测试），可以使用 [pytest-replay](https://github.com/ESSS/pytest-replay) 按相同的顺序重放这些测试，这应该有助于把导致失败的那段序列缩减到最小。

### 测试顺序与重复 {#test-order-and-repetition}

最好把测试重复运行若干次——按顺序、随机或成组地进行——以便发现潜在的相互依赖以及与状态（tear down，清理拆卸）相关的 bug。而单纯的多次重复运行也很有用，能发现一些只有深度学习的随机性才会暴露出来的问题。


#### 重复运行测试 {#repeat-tests}

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

然后把每个测试运行多次（默认 50 次）：

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

脚注：这个插件不能与 `pytest-xdist` 的 `-n` 参数一起使用。它自 2022 年以来也没有过提交，所以应当把它看作功能可用但不再维护，而不是在活跃开发。

脚注：还有另一个插件 [`pytest-repeat`](https://github.com/pytest-dev/pytest-repeat)，它仍在积极维护，但无法重复运行 `unittest.TestCase` 的测试——它们"无论 `--count` 是多少，始终只运行一次"，并且会发出一个警告。所以这两个插件各有权衡：`pytest-flakefinder` 能处理 `unittest` 类但已停滞，`pytest-repeat` 保持更新，但除了那声警告之外会静默地跳过这些测试。如果你的测试套件构建在 `unittest.TestCase` 之上（很多源自 `transformers` 的套件都是如此），这个限制就决定了取舍。


#### 以随机顺序运行测试 {#run-tests-in-a-random-order}

```bash
pip install pytest-random-order
```

重要：只要安装了 `pytest-random-order`，测试就会被自动随机化，不需要任何配置更改或命令行选项。

如前所述，这使得发现耦合测试成为可能——即某个测试的状态会影响另一个测试的状态。当 `pytest-random-order` 安装后，它会打印该次会话所使用的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

这样，如果某个特定序列出了问题，你可以通过加上那个确切的种子来复现它，例如：

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

只有当你使用完全相同的测试列表（或根本不带列表）时，它才会复现那个确切的顺序。一旦你开始手动缩小列表范围，就不能再依赖种子了，而必须手动按它们失败时的确切顺序逐一列出，并用 `--random-order-bucket=none` 告诉 pytest 不要对这些再随机化，例如：

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

要为所有测试禁用打乱顺序：

```bash
pytest --random-order-bucket=none
```

默认隐含的是 `--random-order-bucket=module`，它会在模块级别打乱文件的顺序。它还可以在 `class`、`package`、`global` 和 `none` 级别打乱。完整细节请参阅它的[文档](https://github.com/pytest-dev/pytest-random-order)。

另一种随机化的替代选择是：[`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly)。这个模块的功能和接口非常相似，但它没有 `pytest-random-order` 中可用的那些分桶（bucket）模式。它一旦安装同样会强行生效，存在同样的问题。

### 外观与交互风格的变体 {#look-and-feel-variations}

#### pytest-sugar {#pytest-sugar}

[pytest-sugar](https://github.com/Teemu/pytest-sugar) 是一个改善观感的插件，会添加一个进度条，并即时显示失败的测试及其断言。它在安装后自动启用。

```bash
pip install pytest-sugar
```

要在不使用它的情况下运行测试，请运行：

```bash
pytest -p no:sugar
```

或者卸载它。



#### 报告每个子测试的名称及其进度 {#report-each-sub-test-name-and-its-progress}

通过 `pytest` 对单个或一组测试（先执行 `pip install pytest-pspec`）：

```bash
pytest --pspec tests/test_optimization.py
```

#### 立即显示失败的测试 {#instantly-shows-failed-tests}

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) 会立即显示失败和错误，而不是等到测试会话结束。

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### 用不用 GPU {#to-gpu-or-not-to-gpu}

在具备 GPU 的环境中，要以仅 CPU 模式进行测试，请加上 `CUDA_VISIBLE_DEVICES=""`：

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

或者如果你有多个 GPU，可以指定让 `pytest` 使用哪一个。例如，如果你有 GPU `0` 和 `1`，想只使用第二个 GPU，可以运行：

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

当你想在不同的 GPU 上运行不同任务时，这很有用。

有些测试必须只在 CPU 上运行，有些可以在 CPU、GPU 或 TPU 任意一种上运行，还有些需要多 GPU。以下跳过（skip）装饰器用于设定测试对 CPU/GPU/TPU 的要求：

- `require_torch` - 此测试只在 torch 下运行
- `require_torch_gpu` - 在 `require_torch` 之外还要求至少 1 个 GPU
- `require_torch_multi_gpu` - 在 `require_torch` 之外还要求至少 2 个 GPU
- `require_torch_non_multi_gpu` - 在 `require_torch` 之外还要求 0 或 1 个 GPU
- `require_torch_up_to_2_gpus` - 在 `require_torch` 之外还要求 0 个、1 个或 2 个 GPU
- `require_torch_tpu` - 在 `require_torch` 之外还要求至少 1 个 TPU

我们用下面的表格来描述对 GPU 的要求：


| GPU 数量 | 装饰器                       |
|--------|--------------------------------|
| `>= 0` | `@require_torch`               |
| `>= 1` | `@require_torch_gpu`           |
| `>= 2` | `@require_torch_multi_gpu`     |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus`  |


例如，下面这个测试只有在有 2 个或更多可用 GPU 并且安装了 PyTorch 时才能运行：

```python no-style
from testing_utils import require_torch_multi_gpu

@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

这些装饰器可以叠加使用：

```python no-style
from testing_utils import require_torch_gpu

@require_torch_gpu
@some_other_decorator
def test_example_slow_on_gpu():
```

一些像 `@parametrized` 这样的装饰器会重写测试名称，因此必须把 `@require_*` 这类跳过装饰器列在最后，才能让它们正确工作。下面是一个正确用法的例子：

```python no-style
from testing_utils import require_torch_multi_gpu
from parameterized import parameterized

@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

这个顺序问题在 `@pytest.mark.parametrize` 上不存在，你把它放在前面或后面都仍然可以工作。但它只对非 unittest 的测试有效。

在测试内部：

- 有多少个可用的 GPU：

```python
from testing_utils import get_gpu_count

n_gpu = get_gpu_count()
```


### 分布式训练 {#distributed-training}

`pytest` 无法直接处理分布式训练。如果尝试这么做——子进程不会正常行事，最终会把自己当成 `pytest`，开始循环地运行测试套件。不过，如果派生（spawn）出一个普通进程，由它再派生出多个工作进程并管理 IO 管道，就可以正常工作。

以下是一些使用这种机制的测试：

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/deepspeed/test_deepspeed.py)

要直接跳到执行部分，请在这些测试中搜索 `execute_subprocess_async` 调用，它位于 [testing_utils.py](testing_utils.py) 中。

你至少需要 2 个 GPU 才能看到这些测试实际运行起来：

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

（`RUN_SLOW` 是 HF Transformers 使用的一个特殊装饰器，通常用于跳过重量级测试）

### 输出捕获 {#output-capture}

在测试执行期间，发送到 `stdout` 和 `stderr` 的任何输出都会被捕获。如果某个测试或 setup 方法失败，其对应的被捕获输出通常会连同失败的堆栈跟踪一起显示。

要禁用输出捕获并让 `stdout` 和 `stderr` 正常输出，使用 `-s` 或 `--capture=no`：

```bash
pytest -s tests/utils/test_logging.py
```

要把测试结果以 JUnit 格式输出：

```bash
py.test tests --junitxml=result.xml
```

### 颜色控制 {#color-control}

要让输出不带颜色（例如，白底上的黄字不可读）：

```bash
pytest --color=no tests/utils/test_logging.py
```

### 把测试报告发送到在线 pastebin 服务 {#sending-test-report-to-online-pastebin-service}

从 pytest 9.1.0 起，这个参数位于独立的 `pytest-pastebin` 插件中（`pip install pytest-pastebin`）。

为每个测试失败创建一个 URL：

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

这会把测试运行信息提交给远程的 Paste 服务，并为每个失败提供一个 URL。你可以照常用法选择测试，或者，例如加上 `-x`，如果你只想发送某一个特定的失败。

为整个测试会话的日志创建一个 URL：

```bash
pytest --pastebin=all tests/utils/test_logging.py
```




## 编写测试 {#writing-tests}

大多数时候，把 `pytest` 和 `unittest` 混用在同一个测试套件中是没有问题的。你可以阅读[这里](https://docs.pytest.org/en/stable/how-to/unittest.html)了解这样做时支持哪些特性，但需要记住的关键一点是：大多数 `pytest` 夹具（fixture）都不能用。参数化也不行，不过我们使用用法相似的 `parameterized` 模块。


### 参数化 {#parametrization}

常常需要用不同的参数把同一个测试运行多次。这可以在测试内部完成，但那样就没法只针对其中一组参数运行该测试。

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

现在，默认情况下这个测试会运行 3 次，每次 `test_floor` 的最后 3 个参数都会被赋予参数列表中对应的参数。

你可以只运行 `negative` 和 `integer` 这两组参数：

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

或者，运行除 `negative` 子测试之外的其余全部子测试：

```bash
pytest -k "not negative" tests/test_mytest.py
```

除了刚提到的 `-k` 过滤器，你还可以查出每个子测试的确切名称，并用它们的确切名称运行其中任意几个或全部。

```bash
pytest test_this1.py --collect-only -q
```

它会列出：

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

于是现在你可以只运行这 2 个指定的子测试：

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
```

[parameterized](https://pypi.org/project/parameterized/) 模块对 `unittest` 和 `pytest` 的测试都适用。

不过，如果测试不是 `unittest`，你可以使用 `pytest.mark.parametrize`。

还是同一个例子，这次使用 `pytest` 的 `parametrize` 标记：

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

与 `parameterized` 一样，用 `pytest.mark.parametrize` 也可以在 `-k` 过滤器不够用时，精细控制运行哪些子测试。只不过这个参数化函数会为子测试生成一套略有不同的名称。它们的格式如下：

```bash
pytest test_this2.py --collect-only -q
```

它会列出：

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

于是现在你可以只运行指定的测试：

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

和前面那个例子一样。



### 文件与目录 {#files-and-directories}

在测试中，我们经常需要知道某些东西相对于当前测试文件的位置，而这并不容易，因为测试可能从多个不同的目录被调用，也可能位于深度不同的子目录中。辅助类 `testing_utils.TestCasePlus` 通过整理好所有基本路径并为它们提供便捷的访问器，解决了这个问题：

- `pathlib` 对象（全部是完全解析后的路径）：

    - `test_file_path` - 当前测试文件的路径，即 `__file__`
    - `test_file_dir` - 包含当前测试文件的目录
    - `tests_dir` - `tests` 测试套件的目录
    - `examples_dir` - `examples` 测试套件的目录
    - `repo_root_dir` - 仓库的目录
    - `src_dir` - `src` 的目录（即 `transformers` 子目录所在的目录）

- 字符串化路径——与上面相同，但这些以字符串而不是 `pathlib` 对象的形式返回路径：

    - `test_file_path_str`
    - `test_file_dir_str`
    - `tests_dir_str`
    - `examples_dir_str`
    - `repo_root_dir_str`
    - `src_dir_str`

要开始使用这些，你只需要确保测试位于 `testing_utils.TestCasePlus` 的子类中。例如：

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

如果你不需要通过 `pathlib` 来操作路径，或者只需要字符串形式的路径，你可以随时对 `pathlib` 对象调用 `str()`，或者使用以 `_str` 结尾的访问器。例如：

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
        examples_dir = self.examples_dir_str
```

#### 临时文件与目录 {#temporary-files-and-directories}

使用唯一的临时文件和临时目录对于并行运行测试至关重要，这样测试之间不会互相覆盖数据。我们也希望每个测试创建的临时文件和目录在该测试结束时被删除。因此，使用 `tempfile` 这类能满足这些需求的包非常必要。

不过，在调试测试时，你需要能看到临时文件或临时目录里放了什么，希望知道它的确切路径，而不是每次重跑测试时都被随机化。

辅助类 `testing_utils.TestCasePlus` 最适合用于这类目的。它是 `unittest.TestCase` 的子类，所以我们在测试模块中可以很方便地继承它。

下面是一个使用示例：

```python
from testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

这段代码会创建一个唯一的临时目录，并把 `tmp_dir` 设为它的位置。

- 创建一个唯一的临时目录：

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

`tmp_dir` 将包含所创建的临时目录的路径。它会在测试结束时被自动删除。

- 创建一个你自己指定的临时目录，确保它在测试开始前是空的，并且测试结束后不清空它。

```python
def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

这在调试时很有用：你想监视某个特定目录，并希望确认之前的测试没有在那里留下任何数据。

- 你可以通过直接覆盖 `before` 和 `after` 参数来改变默认行为，从而得到以下某种行为：

    - `before=True`：临时目录总会在测试开始时被清空。
    - `before=False`：如果临时目录已经存在，其中已有的文件会保留在那里。
    - `after=True`：临时目录总会在测试结束时被删除。
    - `after=False`：临时目录在测试结束时总会保持原样。


脚注：为了安全地执行与 `rm -r` 等价的操作，如果使用了显式的 `tmp_dir`，只允许是项目仓库检出目录的子目录，这样就不会误把 `/tmp` 或文件系统中其他重要部分"核平"掉。也就是说，请始终传入以 `./` 开头的路径。

脚注：每个测试都可以注册多个临时目录，它们全部都会被自动删除，除非另有要求。


#### 临时覆盖 sys.path {#temporary-syspath-override}

如果你需要临时覆盖 `sys.path`，例如为了从另一个测试导入，可以使用 `ExtendSysPath` 上下文管理器。示例：


```python
import os
from testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

### 跳过测试 {#skipping-tests}

当发现了一个 bug 并写了一个新测试，但 bug 还没有修复时，这就很有用。为了能把它提交到主仓库，我们需要确保它在 `make test` 期间被跳过。

方法：

-  **skip** 表示：你预期只有在满足某些条件时测试才会通过，否则 pytest 应当完全跳过该测试。常见的例子是在非 Windows 平台上跳过仅限 Windows 的测试，或者跳过依赖当前不可用的外部资源（例如数据库）的测试。

-  **xfail** 表示：你预期某个测试由于某种原因会失败。常见的例子是针对尚未实现的功能或尚未修复的 bug 的测试。当一个被预期失败（用 `pytest.mark.xfail` 标记）的测试实际上通过了，这就是 xpass，并会在测试摘要中被报告。

两者之间一个重要区别是：`skip` 不会运行测试，而 `xfail` 会运行。所以，如果有 bug 的代码会导致某种影响其他测试的不良状态，就不要使用 `xfail`。

#### 实现 {#implementation}

- 下面是无条件跳过整个测试的方法：

```python no-style
@unittest.skip("this bug needs to be fixed")
def test_feature_x():
```

或者通过 pytest：

```python no-style
@pytest.mark.skip(reason="this bug needs to be fixed")
```

或者以 `xfail` 的方式：

```python no-style
@pytest.mark.xfail
def test_feature_x():
```

下面是根据测试内部的检查来跳过测试的方法：

```python
def test_feature_x():
    if not has_something():
        pytest.skip("unsupported configuration")
```

或者跳过整个模块（通过一个 autouse 夹具——`pytest.config` 已在 pytest 4.0 中移除）：

```python
import pytest

@pytest.fixture(scope="module", autouse=True)
def _require_custom_flag(pytestconfig):
    if not pytestconfig.getoption("--custom-flag", default=False):
        pytest.skip("--custom-flag is missing, skipping tests")
```

或者以 `xfail` 的方式：

```python
def test_feature_x():
    pytest.xfail("expected to fail until bug XYZ is fixed")
```

- 下面是当缺少某个导入时跳过模块中所有测试的方法：

```python
docutils = pytest.importorskip("docutils", minversion="0.3")
```

-  根据某个条件跳过测试：

```python no-style
@pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
def test_feature_x():
```

或者：

```python no-style
@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_feature_x():
```

或者跳过整个模块：

```python no-style
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
class TestClass():
    def test_feature_x(self):
```

更多细节、示例和方法见[这里](https://docs.pytest.org/en/stable/how-to/skipping.html)。



### 捕获输出 {#capturing-outputs}

#### 捕获 stdout/stderr 输出 {#capturing-the-stdoutstderr-output}

为了测试会写入 `stdout` 和/或 `stderr` 的函数，测试可以使用 `pytest` 的 [capsys 系统](https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html)来访问这些流。下面是实现方式：

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
    out, err = capsys.readouterr()  # 消费掉被捕获的输出流
    # 可选：如果你想重放已消费掉的这些流：
    sys.stdout.write(out)
    sys.stderr.write(err)
    # 测试：
    assert msg in out
    assert msg in err
```

当然，大多数时候 `stderr` 会作为异常的一部分出现，所以这种情况下必须使用 try/except：

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

捕获 stdout 的另一种方法是通过 `contextlib.redirect_stdout`：

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
    # 可选：如果你想重放已消费掉的这些流：
    sys.stdout.write(out)
    # 测试：
    assert msg in out
```

捕获 stdout 有一个重要的潜在问题：其中可能包含 `\r` 字符，而在正常的 `print` 中，它们会重置此前已经打印的一切。`pytest` 没有问题，但用 `pytest -s` 时这些字符会被包含在缓冲区里，所以为了让测试在使用和不使用 `-s` 时都能运行，你必须对被捕获的输出额外做一次清理，使用 `re.sub(r'~.*\r', '', buf, 0, re.M)`。

不过，我们有一个辅助的上下文管理器包装器，无论其中是否含有 `\r`，它都会自动处理好这一切，所以用起来很简单：

```python
from testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

下面是一个完整的测试示例：

```python
from testing_utils import CaptureStdout

msg = "Secret message\r"
final = "Hello World"
with CaptureStdout() as cs:
    print(msg + final)
assert cs.out == final + "\n", f"captured: {cs.out}, expecting {final}"
```

如果你想捕获 `stderr`，请改用 `CaptureStderr` 类：

```python
from testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

如果你需要同时捕获两个流，请使用父类 `CaptureStd`：

```python
from testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

另外，为了帮助调试测试问题，这些上下文管理器默认会在退出上下文时自动重放被捕获的流。


#### 捕获日志记录器的流 {#capturing-logger-stream}

如果你需要验证某个日志记录器（logger）的输出，可以使用 `CaptureLogger`：

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

### 使用环境变量进行测试 {#testing-with-environment-variables}

如果你想测试环境变量对某个特定测试的影响，可以使用辅助装饰器 `transformers.testing_utils.mockenv`：

```python
from testing_utils import mockenv


class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

有时需要调用外部程序，这要求在 `os.environ` 中设置 `PYTHONPATH`，以包含多个本地路径。辅助类 `testing_utils.TestCasePlus` 可以帮上忙：

```python
from testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # 现在调用外部程序，并把 `env` 传给它
```

根据测试文件是位于 `tests` 测试套件还是 `examples` 套件之下，它会把 `env[PYTHONPATH]` 正确地设置为包含这两个目录之一，同时还会包含 `src` 目录，以确保是针对当前仓库进行测试；最后，如果调用测试之前 `env[PYTHONPATH]` 本来已设有内容，也会一并保留。

这个辅助方法会创建 `os.environ` 对象的一个副本，因此原对象保持不变。


### 获得可复现的结果 {#getting-reproducible-results}

当一个测试需要在各次运行间得到完全相同的结果时，请固定随机数生成器（RNG）以及与 GPU 确定性相关的设置项——在 CUDA 上，只设置随机种子是不够的。完整的做法见[在基于随机性的软件中实现确定性](../training/reproducibility/README.md#achieve-determinism-in-randomness-based-software)；请在测试中调用该章节的 `enforce_reproducibility(seed)`（或它所说明的等效设置），而不是在这里再维护一份不完整的第二副本。

## 调试测试 {#debugging-tests}

要在出现警告的那个位置启动调试器，可以这样做：

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```


## 为生成多份 pytest 报告而做的大规模修补 {#a-massive-hack-to-create-multiple-pytest-reports}

这是我在多年前做的一次大规模的 `pytest` 打补丁（patching），目的是帮助更好地理解 CI 报告。

要启用它，请把下面的内容加到 `tests/conftest.py`（如果你还没有这个文件，就创建一个）：

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

然后在运行测试套件时，像这样加上 `--make-reports=mytests`：

```bash
pytest --make-reports=mytests tests
```

它会生成 8 份独立的报告：
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

所以现在，不再只有来自 `pytest` 的一份把所有内容混在一起的输出，而是每种类型的报告都保存在各自的文件里。

这个功能在 CI 上最为有用，它让定位问题以及查看和单独下载各类报告都容易得多。

针对不同组的测试给 `--make-reports=` 传入不同的值，就可以把每组分别保存，而不会互相覆盖。

这些功能全都本来就在 `pytest` 内部，只是没有办法方便地提取出来，所以我在 [testing_utils.py](testing_utils.py) 里加了 monkey patch 覆盖。嗯，我确实问过能否把这个作为特性贡献给 `pytest`，但我的提议并不受欢迎。
