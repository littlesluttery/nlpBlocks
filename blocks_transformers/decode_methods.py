import math
import torch
import numpy as np
from typing import List, Iterable, Dict, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams["font.sans-serif"]=["FangSong"]  # 指定默认字体 SimHei为黑体 FangSong仿宋
mpl.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

input_ids = torch.tensor([[0,1,2]])  # 输入
scores = torch.tensor([[2.1,-1,-10,1,0]]) # 概率值
scores.shape     # 词表大小

def show_scores(scores1,scores2):
    bar_with = 0.35
    x = torch.arange(len(scores1[0]))
    soft_max_scores1 = torch.nn.functional.softmax(scores1,dim=-1)
    soft_max_scores2 = torch.nn.functional.softmax(scores2,dim=-1)
    plt.bar(x,soft_max_scores1[0],bar_with,align="center",color="#66c2a5",label="处理前")
    plt.bar(x+bar_with,soft_max_scores2[0],bar_with,align="center",color="#8da0cb",label="处理后")
    plt.legend()
    plt.show()

show_scores(scores,scores)

x = np.arange(5)
y = [6,10,4,5,1]
y1 = [2,6,3,8,5]

bar_width = 0.35
tick_label = ["A", "B", "C", "D", "e"]

plt.bar(x, y, bar_width, align="center", color="#66c2a5", label="班级A")
plt.bar(x+bar_width, y1, bar_width, align="center", color="#8da0cb", label="班级B")

plt.xlabel("测试难度")
plt.ylabel("试卷份数")
plt.xticks(x+bar_width/2, tick_label)
plt.legend()

plt.savefig('./tmp.jpg')
plt.show()

def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
from IPython.display import Image
Image(filename="./image/top_k.png",width=400,height=400)

class TopKLogitsWarper():
    def __init__(
        self,
        top_k:int,
        filter_value:float=-float("Inf"),
        min_tokens_to_keep:int=1
    ):
        if not isinstance(top_k,int) or top_k<=0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        # min_tokens_to_keep:搜索序列的个数
        self.top_k = max(top_k,min_tokens_to_keep)
        # 填充值，默认负无穷
        self.filter_value = filter_value

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self,
            input_ids:torch.LongTensor,
            scores:torch.FloatTensor
    ) -> torch.FloatTensor:
        # 防止top_k大于词表
        top_k = min(self.top_k,scores.size(-1)) # 安全检查
        # 先找到概率最大的k个索引
        indices_to_remove = scores < torch.topk(scores,top_k)[0][...,-1,None]
        print(indices_to_remove)
        # 将其他元素填充为负无穷
        scores_processed = scores.masked_fill(indices_to_remove,self.filter_value)
        return scores_processed
processor = TopKLogitsWarper(top_k=3)
scores_pro = processor(input_ids,scores)
print(scores)
print(scores_pro)
show_scores(scores,scores_pro)

from IPython.display import Image
Image(filename="./image/top_p.png",width=400,height=400)

class TopPLogitsWarper():
    def __init__(
        self,
        top_p:float,
        filter_value:float=-float("Inf"),
        min_tokens_to_keep:int=1
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        # 这个可能是历史遗留？外面都已经确认好了
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self,
            input_ids:torch.LongTensor,
            scores:torch.FloatTensor,
    )-> torch.FloatTensor:
        # 先将原始scores按照升序排序
        sorted_logits, sorted_indices = torch.sort(scores,descending=False)
        # 先取softmax，然后计算累计值，cumsum得到一个累加的数值列表
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # 找到累加概率小于(1 - self.top_p)的，结果为[ True,  True,  True,...  True, False, ... False]
        # True 代表要将概率置为负无穷的 token，False 代表要保留的 token
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # 如果最后的 False 少于 min_tokens_to_keep 个，则设置至少保留 min_tokens_to_keep，将后面几个设为 False
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # 将位置为 True 的元素填充负无穷
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed

processor = TopPLogitsWarper(top_p=0.4)
scores_pro = processor(input_ids,scores)
print(scores)
print(torch.nn.functional.softmax(scores,dim=-1))
print(scores_pro)
show_scores(scores,scores_pro)

from IPython.display import Image
Image(filename="./image/min_p.png",width=400,height=400)

class MinPLogitsWarper():
    def __init__(
        self,
        min_p:float,
        filter_value:float=-float("Inf"),
        min_tokens_to_keep:int=1
    ):
        if not(0<=min_p<=1.0):
            raise ValueError(f"`min_p` has to be a float in the [0,1] interval, but is {min_p}")
        if not isinstance(min_tokens_to_keep,int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")
        self.min_p = min_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids:torch.LongTensor,
        scores:torch.FloatTensor
    ) -> torch.FloatTensor:
        # 使用softmax得到概率分布
        probs = torch.softmax(scores,dim=-1)
        print(probs)
        # 选取每个序列概率最高的那个token的概率
        top_probs,_ = probs.max(dim=-1,keepdim=True)
        print(top_probs)
        # 计算截断概率
        scaled_min_p = self.min_p * top_probs
        print(scaled_min_p)
        # 筛选出概率小于截断概率的位置
        tokens_to_remove = probs < top_probs
        print(tokens_to_remove)
        # 按照降序排序，并只得到降序后的索引
        sorted_indices = torch.argsort(scores,descending=True,dim=-1)
        print(sorted_indices)
        # torch.gather按照给的索引(sorted_indices)从tokens_to_remove中取出值。
        sorted_indices_to_remove = torch.gather(tokens_to_remove,dim=-1,index=sorted_indices)
        print(sorted_indices_to_remove)
        # 确保至少有min_tokens_to_keep个会保留
        sorted_indices_to_remove[...,:self.min_tokens_to_keep] = False
        # 将True的填充为负无穷，过滤。
        indices_to_remove = sorted_indices_to_remove.scatter(1,sorted_indices,sorted_indices_to_remove)
        print(indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove,self.filter_value)
        print(scores_processed)
        return scores_processed
processor = MinPLogitsWarper(min_p=0.2)
scores_pro = processor(input_ids, scores)
print(scores)
print(torch.nn.functional.softmax(scores, dim=-1))
print(scores_pro)
show_scores(scores,scores_pro)
class TemperatureLogitsWarper(LogitsProcessor):
    def __init__(
        self,
        temperature:float
    ):
        if not isinstance(temperature,float) or not (temperature>0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
           # 想让temperature == 0.0，可以设置do_sample=False实现相同功能
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)
        self.temperture = temperature

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids:torch.LongTensor,
        scores:torch.FloatTensor
    ) -> torch.FloatTensor:
        # 采样很朴素的实现方式，将原始分数除以温度系数
        scores_processed = scores/self.temperture
        return scores_processed
processor = TemperatureLogitsWarper(temperature=0.7)
scores_pro = processor(input_ids,scores)
print(scores)
print(scores_pro)
show_scores(scores,scores_pro)
class TypicalLogitsWarper():
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        # 检查数值范围
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 先求 H(y_t)
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)  # 对数概率分数 log p(y_t)
        p = torch.exp(normalized)  # 概率分布 p(y_t)
        ent = -(normalized * p).nansum(-1, keepdim=True)  # 信息量 H(y_t), .nansum代表将非 nan 的数字求和

		# 求 ϵ
        shifted_scores = torch.abs((-normalized) - ent)  # ϵ 公式
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)  # 按 ϵ 大小排序并得到索引
        sorted_logits = scores.gather(-1, sorted_indices)  # 按照 ϵ 的大小去取 score 分数
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # 将分数概率化，然后累加筛选出 采样空间词的概率超过阈值

        # 将超过阈值的token分数置为负无穷，使用类似 top_p 的方法
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # 将True的填充为负无穷，即需要过滤掉的
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
processor = TypicalLogitsWarper(mass=0.8)
scores_pro = processor(input_ids, scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class EpsilonLogitsWarper():
    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        epsilon = float(epsilon)
        # 数值范围检查
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 先概率化，然后找到概率小于设定值的
        probabilities = scores.softmax(dim=-1)
        # True的过滤掉
        indices_to_remove = probabilities < self.epsilon

        # 同样的方式，将其置为负无穷
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
    processor = EpsilonLogitsWarper(epsilon=0.2)
scores_pro = processor(input_ids, scores)
print(scores)
print(torch.nn.functional.softmax(scores, dim=-1))
print(scores_pro)
show_scores(scores, scores_pro)

class EtaLogitsWarper():
    def __init__(
        self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, device: str = "cpu"
    ):
        epsilon = float(epsilon)
        # 数值范围检查
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = torch.tensor(epsilon, device=device)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 先概率化
        probabilities = scores.softmax(dim=-1)
        # 计算原始分数的熵
        entropy = torch.distributions.Categorical(logits=scores).entropy()
        # 使用论文中的计算方式得到截止概率
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[..., None]
        # 筛选出小于截止概率的 token
        indices_to_remove = probabilities < eta

        # 相同方式，将其置为负无穷
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed

processor = EtaLogitsWarper(epsilon=0.3)
scores_pro = processor(input_ids, scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class RepetitionPenaltyLogitsProcessor():
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores_processed = scores.scatter(1, input_ids, score)
        return scores_processed

processor = RepetitionPenaltyLogitsProcessor(penalty=2.0)
scores_pro = processor(torch.tensor([[0, 3]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

		# 反向惩罚，先取倒数
        self.penalty = 1 / penalty
        # 提前记录prompt中的id
        self.encoder_input_ids = encoder_input_ids

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 先取出token_id为prompt的原始分数
        score = torch.gather(scores, 1, self.encoder_input_ids)

        # 做惩罚
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
		# 将分数替换回去
        scores_processed = scores.scatter(1, self.encoder_input_ids, score)
        return scores_processed

processor = EncoderRepetitionPenaltyLogitsProcessor(penalty=2.0, encoder_input_ids=torch.tensor([[0, 3]]))
scores_pro = processor(input_ids, scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)
class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]  # batch_size
        cur_len = input_ids.shape[-1]  # 序列长度
        scores_processed = scores.clone()  # 复制出来一份原始分数
        # 这里先得到所有 n-gram，再看当前step的gram之前是否出现过，得到之前出现过的token，具体函数在下面
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        # 将这些token概率之后置为负无穷
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed

def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    # 如果序列长度小于ngram_size，就没必要分gram了
    if cur_len + 1 < ngram_size:
        return [[] for _ in range(num_hypos)]
    # 这里先分gram，这里每个batch返回一个字典，key是n-gram中第一个token_id，value当前gram后面的值
    # 因为可能存在 (21, 15), (21, 51) 这两组gram的情况，第一个token相同，后面token不同，这时候就将后面的token整合放到一个列表中
    # 返回举例：{(151643,): [151643, 151644], (151644,): [8948, 872, 77091]}
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    # 搜索当前step的gram是否在之前出现过，如果出现过就提取出对应value值
    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens

def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    # 初始化一个batch大小的空字典，记录出现过的ngram
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()  # 取出第idx个batch内的input_id
        generated_ngram = generated_ngrams[idx]  # 又取出来一层，空字典
        # 拼接 gen_tokens[0:]、gen_tokens[1:]，zip之后每次取出相邻n个id，比如[0,1], [1,2]...
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])  # 取出前缀作为key，最后一个token作为value
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams

def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    # 提取当前step的前n-1个gram，来确定当前step去抑制哪个token
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])

processor = NoRepeatNGramLogitsProcessor(ngram_size=2)
scores_pro = processor(torch.tensor([[2,3,0,2,3]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                f"`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}"
            )
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        # 这里提前建立好prompt中的n-gram词组
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        scores_processed = scores.clone()
        # 直接搜索当前step出现的n-gram
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
            )
            for hypo_idx in range(num_hypos)
        ]
		# 进行惩罚
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed

processor = EncoderNoRepeatNGramLogitsProcessor(encoder_ngram_size=2, encoder_input_ids=torch.tensor([[2,3,0,2,3]]))
scores_pro = processor(torch.tensor([[2]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class SequenceBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, sequence_bias: Dict[Tuple[int], float]):
        self.sequence_bias = sequence_bias
        # 验证一下输入格式是否正确
        self._validate_arguments()

        # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
        # is infered in the first usage, which inhibits initializing here)
        self.length_1_bias = None
        self.prepared_bias_variables = False

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1 - 准备偏置，只在第一次生成时运行，
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        # 2 - 准备一个空的数组用于存储偏置
        bias = torch.zeros_like(scores)

        # 3 - 先将长度为 1 的偏置加上
        bias += self.length_1_bias

        # 4 - 将长度大于 1 的偏置加上
        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:  # the sequence is of length 1, already applied
                continue
            if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            # 比较input_ids后几个token是否与sequence_ids前几个匹配
            matching_rows = torch.eq(
                input_ids[:, -prefix_length:],
                torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
            ).prod(dim=1)
            bias[:, last_token] += torch.where(
                matching_rows.bool(),
                torch.tensor(sequence_bias, device=input_ids.device),
                torch.tensor(0.0, device=input_ids.device),
            )
        # 5 - 将偏置加到原始分数上
        scores_processed = scores + bias
        return scores_processed
    def _prepare_bias_variables(self, scores: torch.FloatTensor):
    	# 获取词表大小
        vocabulary_size = scores.shape[-1]

        # 检查超过范围的 token_id
        invalid_biases = []
        for sequence_ids in self.sequence_bias:
            for token_id in sequence_ids:
                if token_id >= vocabulary_size:
                    invalid_biases.append(token_id)
        # 如果存在检查超过范围的 token_id，则弹出异常
        if len(invalid_biases) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: "
                f"{invalid_biases}"
            )

        # 将 key 中长度为 1 的 tuple 放入到一个序列中，可以更方便计算
        self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                self.length_1_bias[sequence_ids[-1]] = bias

        self.prepared_bias_variables = True
    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        # 验证是否是字典，字典里面是不是 tuple 等
        if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
            raise ValueError(f"`sequence_bias` has to be a non-empty dictionary, but is {sequence_bias}.")
        if any(not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys()):
            raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
            or len(sequence_ids) == 0
            for sequence_ids in sequence_bias.keys()
        ):
            raise ValueError(
                f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
                f"{sequence_bias}."
            )
        # 偏置都得是 float 型
        if any(not isinstance(bias, float) for bias in sequence_bias.values()):
            raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")

processor = SequenceBiasLogitsProcessor(sequence_bias={(3,0,2): -10.0, (1,): -10.0})
scores_pro = processor(torch.tensor([[0,1,2,3,0]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)


# 继承了SequenceBiasLogitsProcessor类
class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    def __init__(
        self, bad_words_ids: List[List[int]], eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None
    ):
        self.bad_word_ids = bad_words_ids
        # 验证输入参数是否符合要求
        self._validate_arguments()

        # 防止 bad_word_ids 中出现 eos，要将其过滤掉
        if eos_token_id is not None:
            if not isinstance(eos_token_id, torch.Tensor):
                if isinstance(eos_token_id, int):
                    eos_token_id = [eos_token_id]
                eos_token_id = torch.tensor(eos_token_id)

            bad_words_ids = list(
                filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
            )

        # 将某些token设为负无穷，然后父类初始化，就会运行父类的__call__了
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)

    def _validate_arguments(self):
        bad_words_ids = self.bad_word_ids
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )

processor = NoBadWordsLogitsProcessor(bad_words_ids=[[0,3], [1]])
scores_pro = processor(torch.tensor([[0,1,2,3,0]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class SuppressTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, suppress_tokens):
    	# 抑制的 token
        self.suppress_tokens = torch.tensor(list(suppress_tokens))

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 通常使用麻烦的方法修改token分数
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        self.suppress_tokens = self.suppress_tokens.to(scores.device)
        # 选出抑制token的id位置，将其设为True
        suppress_token_mask = torch.isin(vocab_tensor, self.suppress_tokens)
        # 将对应位置设为 -inf
        scores = torch.where(suppress_token_mask, -float("inf"), scores)
        return scores

processor = SuppressTokensLogitsProcessor(suppress_tokens=[0,3])
scores_pro = processor(torch.tensor([[0,1,2,3,0]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class MinLengthLogitsProcessor(LogitsProcessor):
    def __init__(self, min_length: int, eos_token_id: Union[int, List[int], torch.Tensor]):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")
		# 检查eos token
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 实现方法有点怪，直接将索引为eos的分数修改就可以了，弄了这么多
    	# 创建一个vocab大小的顺序索引列表
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        self.eos_token_id = self.eos_token_id.to(scores.device)
        # 将索引是eos的token设为True，其余为False
        eos_token_mask = torch.isin(vocab_tensor, self.eos_token_id)
        scores_processed = scores.clone()
        # 如果长度没达到最小长度要求，就将其置为负无穷
        if input_ids.shape[-1] < self.min_length:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)
        return scores_processed

processor = MinLengthLogitsProcessor(min_length=4, eos_token_id=3)
scores_pro = processor(torch.tensor([[0,0,2,3,0,0]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class ExponentialDecayLengthPenalty(LogitsProcessor):
    def __init__(
        self,
        exponential_decay_length_penalty: Tuple[int, float],
        eos_token_id: Union[int, List[int], torch.Tensor],
        input_ids_seq_length: int,
    ):
		# 输入exponential_decay_length_penalty是一个元组，第一个元素代表从哪个位置开始惩罚，此处加上prompt的长度，代表设置的开始索引是新生成的token数
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        # 第二个代表惩罚因子
        self.regulation_factor = exponential_decay_length_penalty[1]
		# 整理eos
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 获取当前长度
        cur_len = input_ids.shape[-1]
        self.eos_token_id = self.eos_token_id.to(scores.device)
        # 初始化一个值为零的向量
        penalties = torch.zeros_like(scores)
        scores_processed = scores
        # 长度大于设置值后开始惩罚
        if cur_len > self.regulation_start:
            # 获取惩罚的数值，代表从惩罚位置开始又生成了几个token
            penalty_idx = cur_len - self.regulation_start
            # 这里先取eos的绝对值，再乘上长度，防止负增益
            penalty = torch.abs(scores[:, self.eos_token_id]) * (pow(self.regulation_factor, penalty_idx) - 1)
            # 将eos位置的分数替换
            penalties[:, self.eos_token_id] = penalty
            # 加到原始分数上
            scores_processed = scores + penalties
        return scores_processed
processor = ExponentialDecayLengthPenalty(exponential_decay_length_penalty=(2, 1.2), eos_token_id=3, input_ids_seq_length=2)
scores_pro = processor(torch.tensor([[0,0,2,3,0,1,1]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
    	# 提前传入约束函数
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 使用mask的方式选择遮蔽token
        mask = torch.full_like(scores, -math.inf)
        # 一个批次一个批次处理
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
        	# 束搜索处理
            for beam_id, sent in enumerate(beam_sent):
            	# 得到允许生成的token
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                # 如果返回为空，弹出异常
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        # 不被允许生成的token分数加上负无穷
        scores_processed = scores + mask
        return scores_processed

def prefix_allowed_tokens_fn(batch_id, input_ids):
	if input_ids[-1] == 1:
		return [2]
	elif input_ids[-1] == 2:
		return [3]
	return list(range(len(scores[0])))

processor = PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, num_beams=1)
scores_pro = processor(torch.tensor([[0,0,2,3,3]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, bos_token_id: int):
    	# 指定第一个生成的 token_id
        self.bos_token_id = bos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 获取生成token的长度
        cur_len = input_ids.shape[-1]
        scores_processed = scores
        # 只有长度为1时触发，但decoder-only模型一般都有prompt，很难触发
        if cur_len == 1:
            scores_processed = torch.full_like(scores, -math.inf)
            scores_processed[:, self.bos_token_id] = 0
        return scores_processed

processor = ForcedBOSTokenLogitsProcessor(bos_token_id=2)
scores_pro = processor(torch.tensor([[3,2]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)

class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, max_length: int, eos_token_id: Union[int, List[int], torch.Tensor]):
        self.max_length = max_length
		# 整理eos格式
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)
        self.eos_token_id = eos_token_id
		# 验证eos数值
        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 获取生成token的长度
        cur_len = input_ids.shape[-1]
        self.eos_token_id = self.eos_token_id.to(scores.device)
        scores_processed = scores
        # 只在达到最大长度时触发
        if cur_len == self.max_length - 1:
            scores_processed = torch.full_like(scores, -math.inf)
            scores_processed[:, self.eos_token_id] = 0
        return scores_processed

processor = ForcedEOSTokenLogitsProcessor(max_length=6, eos_token_id=2)
scores_pro = processor(torch.tensor([[0,0,2,3]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    def __init__(self, begin_suppress_tokens, begin_index):
    	# 设置某些token需要在第一步抑制
        self.begin_suppress_tokens = torch.tensor(list(begin_suppress_tokens))
        # 提前记录第一步token的序列长度
        self.begin_index = begin_index

	# 没有用到
    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    	# 先初始化词表大小的顺序列表
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        self.begin_suppress_tokens = self.begin_suppress_tokens.to(scores.device)
        # 将需要抑制的token选出来，对应位置设为True
        suppress_token_mask = torch.isin(vocab_tensor, self.begin_suppress_tokens)
        scores_processed = scores
        # 如果是第一步，就将设置的token分数设为负无穷
        # 这一步应该放在最开始，要不每个step都要运行上面代码初始化变量，浪费时间
        if input_ids.shape[-1] == self.begin_index:
            scores_processed = torch.where(suppress_token_mask, -float("inf"), scores)

        return scores_processed

processor = SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens=[3,1], begin_index=5)
scores_pro = processor(torch.tensor([[0,0,2,3,1]]), scores)
print(scores)
print(scores_pro)
show_scores(scores, scores_pro)


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
    	# 检查是不是生成第一个token
        if self.unconditional_context["first_pass"]:
        	# 如果是第一个 token，根据 input_ids 去掉提示词部分，作为无提示生成的输入
        	# 这里整理 input_ids 和 attention_mask
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            # 判断是不是生成第一个 token 的标志位
            self.unconditional_context["first_pass"] = False
        else:
        	# 如果不是生成第一个token，则使用之前的结果
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            # 这里也使用了kv-cache，所以只取了 input_ids 最后一个，跟主模型同样的原理
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

		# 无提示词模型生成一份结果
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

		# 将结果返回
        return out.logits
    def __call__(self, input_ids, scores):
    	# 主模型的原始分数概率化
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        # 这里属于冗余判断了，如果等于1就不会进来这里
        if self.guidance_scale == 1:
            return scores

		# 得到无提示的输出原始分数
        logits = self.get_unconditional_logits(input_ids)
		# 当前 step 的 token 同样概率化
        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        # 使用论文中的计算方式，将两者的差异缩放后补偿到无提示生成结果上
        scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return scores_processed

class HammingDiversityLogitsProcessor(LogitsProcessor):
    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
    	# 检查参数格式
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        # 每个组内的束搜索分支数量
        self._num_sub_beams = num_beams // num_beam_groups
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        # 长度为 num_beams * batch，前 _num_sub_beams 个代表第1组内的分支选择的token，以此类推
        # 注意多个batch数据会拼接到一起，所以 num_beams + _num_sub_beams 索引的数据是第二个batch token的开始
        batch_size = current_tokens.shape[0] // self._num_beams
        # beam_group_idx为当前是第几组，这里找到当前组current_tokens开始的索引
        group_start_idx = beam_group_idx * self._num_sub_beams
        # 找到结束的索引，这里认为如果 num_beams 不能整除 num_beam_groups 的时候，会超出范围，所以做了min判断
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

		# 如果是第一组，自然不用做惩罚
        if group_start_idx == 0:
            return scores

        scores_processed = scores.clone()
        # 批次内挨个处理
        for batch_idx in range(batch_size):
            # 得到前面组的token
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            # 得到vocab_size大小的序列，前面选中的token_id位置 置为该token出现的次数
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            # 将前面选中的token根据数量惩罚 _diversity_penalty * token_frequency 分数
            scores_processed[batch_idx * group_size : (batch_idx + 1) * group_size] -= (
                self._diversity_penalty * token_frequency
            )

        return scores_processed

class InfNanRemoveLogitsProcessor(LogitsProcessor):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # nan 设置为 0
        scores_processed = torch.where(scores != scores, 0.0, scores)

        # +/-inf 设置成当前数据类型的最大/最小值
        scores_processed = torch.where(scores == float("inf"), torch.finfo(scores.dtype).max, scores_processed)
        scores_processed = torch.where(scores == -float("inf"), torch.finfo(scores.dtype).min, scores_processed)

        return scores_processed
processor = InfNanRemoveLogitsProcessor()
scores_pro = processor(torch.tensor([[0,0,2,3,1]]), torch.tensor([[3.1, 1, 0, torch.nan, -torch.inf]]))
print(torch.tensor([[3.1, 1, 0, torch.nan, -torch.inf]]))
print(scores_pro)
show_scores(scores, scores_pro)
class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        vocab_size,
        device,
        greenlist_ratio: float = 0.25,
        bias: float = 2.0,
        hashing_key: int = 15485863,
        seeding_scheme: str = "lefthash",
        context_width: int = 1,
    ):
        if seeding_scheme not in ["selfhash", "lefthash"]:
            raise ValueError(f"seeding_scheme has to be one of [`selfhash`, `lefthash`], but found {seeding_scheme}")
        if greenlist_ratio >= 1.0 or greenlist_ratio <= 0.0:
            raise ValueError(
                f"greenlist_ratio has be in range between 0.0 and 1.0, exclusively. but found {greenlist_ratio}"
            )

        self.vocab_size = vocab_size
        # 按比例得到应该选择的绿色token数量
        self.greenlist_size = int(self.vocab_size * greenlist_ratio)
        self.bias = bias
        self.seeding_scheme = seeding_scheme
        self.rng = torch.Generator(device=device)
        self.hash_key = hashing_key
        self.context_width = context_width

        self.rng.manual_seed(hashing_key)
        self.table_size = 1_000_003
        self.fixed_table = torch.randperm(self.table_size, generator=self.rng, device=device)
    def set_seed(self, input_seq: torch.LongTensor):
    	# 取出窗口大小的token
        input_seq = input_seq[-self.context_width :]
		# 不同算法使用不同的hash方式得到种子数
        if self.seeding_scheme == "selfhash":
        	# selfhash 利用的当前词和窗口词
            a = self.fixed_table[input_seq % self.table_size] + 1
            b = self.fixed_table[input_seq[-1] % self.table_size] + 1
            seed = (self.hash_key * a * b).min().item()
        else:
        	# lefthash 只利用最后一个token
            seed = self.hash_key * input_seq[-1].item()
        self.rng.manual_seed(seed % (2**64 - 1))

    def _get_greenlist_ids(self, input_seq: torch.LongTensor) -> torch.LongTensor:
    	# 根据输入序列确定随机种子
        self.set_seed(input_seq)
        # 按这个随机种子随机排序词表，得到词表大小但乱序的 token_id
        vocab_permutation = torch.randperm(self.vocab_size, device=input_seq.device, generator=self.rng)
        # 按照比例取前面的token作为绿色token名单
        greenlist_ids = vocab_permutation[: self.greenlist_size]
        return greenlist_ids

    def _score_rejection_sampling(self, input_seq: torch.LongTensor, scores: torch.FloatTensor) -> torch.LongTensor:
        """
        Generate greenlist based on current candidate next token. Reject and move on if necessary.
        Runs for a fixed number of steps only for efficiency, since the methods is not batched.
        """
        final_greenlist = []
        # 将当前 step 所有 token 的分数按分数大小降序排列，得到排序之后的索引值
        _, greedy_predictions = scores.sort(dim=-1, descending=True)

        # 40是个任意数字，使用概率最大的前40个token处理水印
        for i in range(40):
        	# 每个token都有一个自己的绿色token名单，这里将输入序列和当前选中的token拼接输入
            greenlist_ids = self._get_greenlist_ids(torch.cat([input_seq, greedy_predictions[i, None]], dim=-1))
            # 如果当前token在名单中，将其加入到候选里
            if greedy_predictions[i] in greenlist_ids:
                final_greenlist.append(greedy_predictions[i])
        return torch.tensor(final_greenlist, device=input_seq.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 序列长度要大于窗口大小
        if input_ids.shape[-1] < self.context_width:
            logger.warning(
                f"`input_ids` should have at least `{self.context_width}` tokens but has {input_ids.shape[-1]}. "
                "The seeding will be skipped for this generation step!"
            )
            return scores
		# 克隆一份分数
        scores_processed = scores.clone()
        # 批次内一个序列一个序列处理
        for b_idx, input_seq in enumerate(input_ids):
        	# 两种处理算法
            if self.seeding_scheme == "selfhash":
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            # 得到候选token之后，将这些token的分数加上bias偏差
            scores_processed[b_idx, greenlist_ids] = scores_processed[b_idx, greenlist_ids] + self.bias

        return scores_processed



















