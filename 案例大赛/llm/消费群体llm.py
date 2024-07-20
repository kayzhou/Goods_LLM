from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import os

start_time = time.time()

model_id = "/home/llm/liguanqun/llama3/model/NousResearch-Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# 函数：保存最新的商品索引
def save_last_processed_index(file_path, index):
    with open(file_path, 'w') as f:
        f.write(str(index))

# test_label = 1
model_name = "llama3"
task = "consumer"
json_file_path = f"/home/llm/liguanqun/案例大赛/data/consumer_2000_test.json"
output_filename = f'/home/llm/liguanqun/案例大赛/results/{model_name}_{task}_analysis_results_test.jsonl'
processed_index_file = f'/home/llm/liguanqun/案例大赛/results/{model_name}_{task}_processed_index_test.txt'
# metrics_output_filename = f'/home/llm/liguanqun/llama3/data/{model_name}/{model_name}_{task}_metrics_results_test_{text_label}.json'

# 读取上次处理的商品索引
try:
    with open(processed_index_file, 'r') as f:
        processed_index = int(f.read())
except FileNotFoundError:
    processed_index = 0

# 存储真实标签和预测标签
# true_labels = []
pred_labels = []

# 循环调用接口并保存结果到txt文件
with open(output_filename, 'a', encoding='utf-8') as outfile:
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        for i, row in enumerate(data):
            if i < processed_index:
                continue
            input_data = json.loads(row['input'])
            # true_label = json.loads(row['output'])

            messages = [
                {"role": "system", "content": row['instruction']},
                {
                    "role": "user",
                    "content": "依托天猫、淘宝平台的海量消费者数据，阿里把当前正在崛起的新消费群体分为特征鲜明的八大人群。你了解是哪八大人群吗？"
                },
                {
                    "role": "assistant",
                    "content": f"""
                                    "分别是：小镇青年、Gen Z（Z世代）、都市白领、精致妈妈、都市中产、都市蓝领、都市银发和小镇中老年。
                                    并且这八大人群还可以进一步划分为三种类型人群，分别为新势力、主力军和蓝海人群。
                                    其中，新势力人群包括小镇青年和Gen Z，他们关注新鲜事物和潮流趋势，对新品接受度比较高，在美妆和个护方面，新势力已经接近中坚力量；
                                    主力军人群包括都市白领、都市中产和精致妈妈，他们对品质化的要求不断提升。
                                    蓝海人群包括都市银发、都市蓝领和小镇中老年，他们追求极致性价比，在购买商品时会表现出少量多次的购买习惯，喜欢活动拼团，更加容易接受简单直接的促销方式。"
                                """
                },
                {
                    "role": "user",
                    "content": "新势力人群中的小镇青年和Gen Z分别有什么特点吗？"
                },
                {
                    "role": "assistant",
                    "content": f"""
                                    "小镇青年是生活在四线城市以下的18-35岁的人群，他们紧追都市潮流，在美食、美妆、电子产品等方面的兴趣偏好显著。重视社交生活，经常与好友拼单；
                                    对价格敏感，注重性价比。他们有充足的闲暇时间通过短视频、直播、网络游戏等方式进行休闲娱乐。并且小镇青年对国家认同感很强，非常愿意购买国产品牌。
                                    Gen Z（Z世代）多居住在一二三线城市，以18-24岁为主，他们倾向于数字世界中进行社交、学习和娱乐，对新奇有趣的事物充满热情。Gen Z特别在意外表，是潮流服饰的忠实拥趸。
                                    此外，他们热衷于利用互联网发展自己的兴趣圈子（如宅文化、二次元、电竞等），在短视频、直播平台上进行小众社交。"
                                """
                },
                {
                    "role": "user",
                    "content": "主力军人群中的都市白领、都市中产和精致妈妈分别有什么特点吗？"
                },
                {
                    "role": "assistant",
                    "content": f"""
                                    "都市白领居住在一二三线城市，以25-35岁为主，家庭月收入在6-25k，他们仍然处于事业奋斗期，对消费便利性要求高，购物热情旺盛，乐于尝试新鲜事物；
                                    并对提升自我价值十分关注，是护肤美妆、知识付费等消费的主力人群。
                                    都市中产主要生活在一二三线城市，以70后、80后为主，他们时常以家庭为单位进行社交，比如周末朋友小酌、孩子生日派对等等。相比于年轻一代，他们持有更加理性的消费观。
                                    并且在衣食住行方面的消费全面升级，比如给自己购买高端的潮流服饰，给家人购买进口的保健食品。他们大多数持以“轻价格，重品质”的消费理念。
                                    精致妈妈是指处于孕期到小孩12岁以内的女性，多生活在一二三线城市。她们愿意花钱买便利，热衷线上购物。她们经常浏览宝宝树等母婴社区与小红书等社交电商，分享购物与使用心得。
                                    在母婴用品和小家电等方面的兴趣偏好较为显著。并且值得注意的是，精致妈妈们非常重视产品的健康与安全，青睐海淘渠道购买海外生产的高品质婴儿食品与用品。"
                                """
                },
                {
                    "role": "user",
                    "content": "蓝海人群中的都市银发、都市蓝领和小镇中老年分别有什么特点吗？"
                },
                {
                    "role": "assistant",
                    "content": f"""
                                    "年龄在50岁以上的都市银发一族，大多生活在一二三线城市，对高品质养老生活需求较为明显。他们非常关注自身健康状况，热爱购买保健食品、健康检测产品等等。
                                    由于根深蒂固的节省消费观，他们线上购物时追求性价比，偏爱家庭装折扣产品。他们重视家人和朋友关系的维护，时常和老朋友们在家喝点小酒、下下象棋等。
                                    都市蓝领主要为25-50岁的人群，他们在高线城市自强自立，大多居住在城市郊区。他们收入偏低，生活成本高，购物时很在意产品性价比。
                                    因为地铁、公交通勤时间较长，仅有的消遣就是每个月给视频App、游戏账户充值会员。他们在家清百货和虚拟服务等方面的兴趣偏好较为显著。
                                    小镇中老年生活在四线及以下城市，年龄为大于35岁的人群，生活节奏较慢。他们喜欢向身边群体分享健康、时事、购物等各种信息，线上消费偏低，以跟随型消费为主。
                                    他们逐渐学会在跨境平台上买些高品质的进口产品给经常回家的孩子们，偶尔也会买来自我享受一下。极致的性价比对他们来说依旧非常重要。在药品和运动服饰等方面的兴趣偏好较为显著。"
                                """
                },
                {
                    "role": "user",
                    "content": "可以依据上述描述分别总结一下八大消费群体较为偏好的商品品类需求吗？"
                },
                {
                    "role": "assistant",
                    "content": f"""
                                    "总结如下：
                                    小镇青年：食品、美妆、电子产品，愿意购买国产品牌，对价格敏感，注重性价比；
                                    Gen Z：知识付费、潮流服饰、虚拟服务，特别在意穿着造型，对新生事物充满热情；
                                    都市白领：护肤美妆、知识付费，对新品牌接纳程度高，注重提升自我价值；
                                    精致妈妈：母婴用品、小家电、家清百货，重视产品的健康与安全；
                                    都市中产：潮流服饰、保健食品、高品质商品，消费较为理性，注重个人品味；
                                    都市蓝领：家清百货、虚拟服务、个人洗护，注重产品性价比；
                                    都市银发：酒类、保健食品、健康监测设备，愿意过上高品质养老生活注重性价比，偏爱家庭装折扣产品；
                                    小镇中老年：药品、运动服饰，偶尔会买高品质进口产品，注重性价比。
                                    "
                                """
                },
                {
                    "role": "user", 
                    "content": f""" 
                                    "请根据下列商品名称和商品的中位数价格对该商品进行消费群体分类，按顺序输出该商品最有可能属于的三种消费人群。消费群体类别从上述八大消费群体以及他们所属的类型中选取。
                                    商品名称: 1.25L*2瓶厨邦金品生抽, 商品中位数价格: 19.9元" 
                                """
                },
                {"role": "assistant", "content": "{\"消费群体1\": \"都市蓝领\", \"消费群体1所属类型\": \"蓝海人群\", \"消费群体2\": \"精致妈妈\", \"消费群体2所属类型\": \"主力军人群\", \"消费群体3\": \"小镇中老年\", \"消费群体3所属类型\": \"蓝海人群\"}"},
                {"role": "user", "content": f"商品名称: {input_data['商品名称']}, 商品中位数价格: {input_data['商品中位数价格']}元"}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.25,
                top_p=0.2,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            response = outputs[0][input_ids.shape[-1]:]
            response_str = tokenizer.decode(response, skip_special_tokens=True)

            # 记录真实标签和预测标签
            # true_labels.append(true_label)
            try:
                pred_label = json.loads(response_str)
            except json.JSONDecodeError:
                pred_label = {}
            pred_labels.append(pred_label)

            # 将原始标签和预测标签写入jsonl文件
            output_line = {
                "goods_name": json.dumps(input_data['商品名称'], ensure_ascii=False),
                "median_price": json.dumps(input_data['商品中位数价格'], ensure_ascii=False),
                # "label": json.dumps(true_label, ensure_ascii=False),
                "predict": json.dumps(pred_label, ensure_ascii=False)
            }
            print(f"第{i + 1}个")
            print(output_line)
            outfile.write(json.dumps(output_line, ensure_ascii=False) + '\n')
            outfile.flush()

            save_last_processed_index(processed_index_file, processed_index + i + 1)  # 保存最新的商品索引


end_time = time.time()
execution_time = end_time - start_time

# 打印程序运行时间
print(f"程序运行时间：{execution_time} 秒")
