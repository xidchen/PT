# !pip install cpm_kernels sentencepiece transformers
# It works on Colab T4

import transformers


model_path = "THUDM/chatglm3-6b"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
).half().cuda()
model = model.eval()


def extract_quotes(s: str) -> str:
    """Extract quotes between parentheses and after closing parenthesis"""
    quotation_marks = {"\"", "\'"}
    for qm in quotation_marks:
        if s.count(qm) == 2:
            left_qm_index = s.index(qm)
            right_qm_index = s.index(qm, left_qm_index + 1)
            if s.endswith(qm):
                s = s[left_qm_index + 1:right_qm_index]
            else:
                s = s[left_qm_index + 1:right_qm_index] + s[right_qm_index + 1:]
        while qm in s:
            s = s.replace(qm, __new="")
    return s


def extract_chinese_quotes(s: str) -> str:
    """Extract quotes between chinese parentheses and after closing parenthesis"""
    left_qm, right_qm = "“", "”"
    left_qm_index = s.index(left_qm) if left_qm in s else - 1
    right_qm_index = s.index(right_qm) if right_qm in s else len(s)
    if s.endswith(right_qm):
        s = s[left_qm_index + 1:right_qm_index] + s[right_qm_index + 1:]
    else:
        s = s[left_qm_index + 1:right_qm_index]
    return s


def process_bracket(s: str) -> str:
    """Remove contents in brackets"""
    left_bkt, right_bkt = "(", ")"
    if left_bkt in s or right_bkt in s:
        while left_bkt in s and right_bkt in s:
            left_bkt_index = s.index(left_bkt)
            right_bkt_index = s.index(right_bkt, left_bkt_index + 1)
            s = s[:left_bkt_index] + s[right_bkt_index + 1:]
        if (left_bkt in s) == (right_bkt not in s):
            s = ""
    return s


def process_chinese_bracket(s: str) -> str:
    """Remove chinese contents in brackets"""
    left_bkt, right_bkt = "（", "）"
    if left_bkt in s or right_bkt in s:
        while left_bkt in s and right_bkt in s:
            left_bkt_index = s.index(left_bkt)
            right_bkt_index = s.index(right_bkt, left_bkt_index + 1)
            s = s[:left_bkt_index] + s[right_bkt_index + 1:]
        if (left_bkt in s) == (right_bkt not in s):
            s = ""
    return s


def process_colon(s: str) -> str:
    """Extract content after colon unless exception"""
    colon = "："
    while colon in s:
        colon_index = s.index(colon)
        s = s[colon_index + 1:]
    return s


def remove_carriage_return(s: str) -> str:
    """Remove carriage return"""
    s = s.replace("\n", "").replace("\r", "")
    return s


def deduplicate_input(s: str, s_input: str) -> str:
    """Deduplicate input from the output"""
    if s_input in s:
        s = s.replace(s_input, "")
    return s


def remove_starting_punctuation(s: str) -> str:
    """Remove starting punctuations from the output"""
    punctuations = {"，", ",", "。", "！", "？"}
    if s and s[0] in punctuations:
        s = s[1:].strip()
    return s


def remove_hashtag_loop(s: str) -> str:
    """remove hashtags until no hashtags"""
    hashtag = "#"
    while hashtag in s:
        s = remove_one_hashtag(s)
    return s


def remove_one_hashtag(s: str) -> str:
    """Remove one hastag at a time"""
    hashtag = "#"
    exception_chars = {"，", "@"}
    if hashtag in s:
        hashtag_index = s.index(hashtag)
        s_before_hashtag = s[:hashtag_index]
        for char_index in range(hashtag_index, len(s)):
            if s[char_index] in exception_chars:
                s_after_hashtag = s[char_index:]
                s = s_before_hashtag + s_after_hashtag
                return s
        s = s_before_hashtag
    return s


def customize_output(s: str) -> str:
    """Customize generated output, and remove redundant characters"""
    s = extract_quotes(s)
    s = extract_chinese_quotes(s)
    s = process_bracket(s)
    s = process_chinese_bracket(s)
    s = process_colon(s)
    s = remove_carriage_return(s)
    s = remove_starting_punctuation(s)
    s = remove_hashtag_loop(s)
    return s


def produce_response(tknzer, request) -> str:
    response, history = model.chat(tknzer, request, history=[])
    response_word_limit = [1, 100]
    while (
        not isinstance(response, str) or
        len(response) <= response_word_limit[0] or
        len(response) >= response_word_limit[1]
    ):
        response, history = model.chat(tknzer, request, history=[])
    return response


def customize_prompt(acc: str, cs: str, s_1: str, s_2: str) -> str:
    """Customize prompt by inserting a few strings"""
    prompt = (
        f"请以一个普通人的角色以视频文案为背景，"
        f"针对评论内容回复表示积极真诚又相互鼓励内容的一个简短的句子，"
        f"越短越好有创意，和评论内容产生共鸣。不要重复视频文案。不要重复评论内容。"
        f"{acc}全中文。{cs}视频文案是\"{s_1}\"，评论内容是\"{s_2}\"。"
    )
    return prompt


def fail_final_check(s: str) -> bool:
    """Check whether the response fails to pass the final check"""
    fail, not_fail = True, False
    if fail_language_check(s):
        return fail
    if fail_punctuation_check(s):
        return fail
    return not_fail


def fail_language_check(s: str, languages=None) -> bool:
    """Check whether the response fails to pass the language check"""
    if languages is None:
        languages = {"zh"}
    fail, not_fail = True, False
    if languages == {"zh"}:
        for char in s:
            if is_cjk_character(char):
                return not_fail
    return fail


def is_cjk_character(char: str) -> bool:
    """Check whether a character is a CJK character"""
    return True if int(0x4e00) <= ord(char) <= int(0x9fff) else False


def fail_punctuation_check(s: str) -> bool:
    """Check whether the response fails to pass the punctuation check"""
    fail, not_fail = True, False
    wrong_combinations = {"，？", "@！"}
    for combination in wrong_combinations:
        if combination in s:
            return fail
    return not_fail


def print_responses(s_acc, cs, s_1, s_2_list):
    """Print responses by providing acc and a sentence and a list of sentences"""
    reproduction_number = 1
    for s_2 in s_2_list:
        request = customize_prompt(s_acc, cs, s_1, s_2)
        for _ in range(reproduction_number):
            response = produce_response(tokenizer, request)
            response = customize_output(response)
            while fail_final_check(response):
                response = produce_response(tokenizer, request)
                response = customize_output(response)
            print(response)
        print()


acc_content = ""
cs_contempt = ""
video_content = (
    "自己不努力，没人会帮你，走好自己的路。不管什么年纪，别辜负最好的自己，"
    "你所浪费的今天，是许多人奢望的明天，无论你正在经历什么，都要调整好心态，砥砺前行"
    "#坚持#赚钱#成功#努力#生活#励志#正能量#递梦驿站"
)
comment_content_list = [
    "靠自己！ 加油🥰",
    "感恩🙏 好",
    "可是好累😭...",
    "要努力加油 才會苦盡甘來🥰",
    "沒錯～說的太好了",
    "努力不是为了超越别人，而是给自己一个交代。美好的东西从不轻易获得，成功靠的不是豪言壮语，而是脚踏实地的努力。",
    "是的，不停的提升自己",
    "对，加油👍👍👍👍👍👍👍👍👍👍👍👍👍👍👍👍👍👍",
    "對不能靠誰只能靠自己",
]

print_responses(acc_content, cs_contempt, video_content, comment_content_list)


acc_content = "偶尔可以在结尾留下联系方式@{}，让对方关注".format("8237ak")
cs_content = ""
video_content = "就差你了#赚钱#赚钱的方法#灰产偏门#兼职副业#偏门项目#捞偏门"
comment_content_list = [
    "日四位数，免费带",
    "求带",
    "免费带你",
    "有人能带带我吗？",
]

print_responses(acc_content, cs_content, video_content, comment_content_list)


acc_content = "偶尔可以在结尾留下自己联系方式@{}，引起对方关注".format("iuwqi")
cs_content = ""
video_content = (
    "Share a zero-cost profitable project suitable for Malaysians. "
    "零成本创业项目分享，适合🇲🇾的朋友。#赚钱 #副业 #商业思维 #游戏推广 #tiktok赚钱"
)
comment_content_list = [
    "你听过Amway 吗",
    "有项目吗 求带",
    "please 啦 ！！！ 如果能,你老早就变大老板 还傻傻在这里拍抖音TikTok 😂😂😂",
]

print_responses(acc_content, cs_content, video_content, comment_content_list)


acc_content = "偶尔可以在结尾留下自己联系方式@{}，引起对方关注".format("ua090")
cs_content = ""
video_content = "在股市种经典实用的主力解套战法！快收藏起来！#股市 #投资 #炒股 #干货分享 #财经"
comment_content_list = [
    "纸上谈兵",
    "然后一路跌到2元，再也没有明显上涨",
    "你要有w你懂10w 可能是一股而以...",
    "通常一賣就漲停",
    "跌穿了怎么办",
    "想的挺美",
    "😁当你卖又漲回去了 你买进了又跌了",
    "請問幾點睡可以做這麼美的夢",
    "😅當年中車我也是這樣操作，結果。。。唉。。。",
    "股票不配合怎麼辦",
    "对于那些想要了解金融知识的人来说，你的视频是一个很好的资源！",
]

print_responses(acc_content, cs_content, video_content, comment_content_list)


acc_content = "偶尔可以在结尾留下自己联系方式@{}，引起对方关注".format("akl329")
cs_content = ""
video_content = "你还在打工的路上吗？越是穷越是困难的时候，越要创业 #创业#创业项目#商业#生意#商业模式#商业思维"
comment_content_list = [
    "20年前确实行得通！可现在是21世纪了 兄弟！",
    "没有客源 一切免谈。没有订单 也免谈。",
    "说很容易，做就难了",
    "理论上可以，现实根本行不通，",
    "说当然简单😂",
    "事实上能赚百分之三十的已经算是利润很高的了",
    "要会做。先打工",
]

print_responses(acc_content, cs_content, video_content, comment_content_list)
