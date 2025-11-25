现在的方案是针对每条数据生成 key aspects，随后和数据一起送入 LLM Judge。但是现有的方法效果对比原版 reward bench 效果有限。

原因分析：原 rewardbench 拥有强大细致的 prompt，而我们的 prompt 虽然具有一定的方法创新（自我提示），但 key aspects 的 3-5 个方面对于整个 prompt 而言占比过于小了。

下面提出针对性的修改解决方案。

1. 把生成 key aspects 换成生成一个 evaluation plan，该方法受论文文件： Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge.pdf 的启发，你可以查阅，我们目前仅使用论文中的 prompt。按照 figure 3：（这里似乎没给模型看待评判的成对回复内容，如果可以的话在代码中用一个可以注释的行开关回复内容的加入，以手动比较两者。）

   We want to evaluate the quality of the responses provided by AI assistants to the user question displayed below. For that, your task is to help us build an evaluation plan that can then be executed to assess the response quality. Whenever appropriate, you can choose to also include a step-by-step reference answer as part of the evaluation plan. Enclose your evaluation plan between the tags “[Start of Evaluation Plan]” and “[End of Evaluation Plan]”.

   [User Question]
   {instruction}

   为了适配我们的生成代码，你还需要在 prompt 中加入适配各数据集的评估领域，比如 rewardbench 中的 section，另两个数据集同理（目前的代码也有加入）。

2. 然后在 self-synthesized 的评估代码中改用下面的 prompt 评估（来自上述论文的 figure 4）：

   Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should be performed by following the provided evaluation plan step-by-step. Avoid copying the plan when doing the evaluation. Please also only stick to the given plan and provide explanation of how the plan is executed to compare the two responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your evaluation, output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better.

   [User Question]
   {instruction}

   [The Start of Assistant A’s Answer]
   {response A}
   [The End of Assistant A’s Answer]

   [The Start of Assistant B’s Answer]
   {response B}
   [The End of Assistant B’s Answer]

   [The Start of Evaluation Plan]
   {evaluation plan}
   [The End of Evaluation Plan]

   这样就可以增大评估计划在整个 prompt 中的比重，使其发挥显著的作用。

下面说明具体更改要求：

1. 在 datasets 中现在有三个数据集和对应的预处理脚本，刚才提出的方案要求你仅在 rewardbench 数据集进行测试，其他两个数据集先不动，测试完之后我会要求你应用到剩余数据集的处理中。处理脚本的名字改成 generate_evaluation_plan.py 和 run_evaluation_plan.sh，生成的数据中，"key_focus_aspects"字段替换成我们现在的 "evaluation_plan" 字段。
2. 数据集处理全集太费钱，现在需要你对要测试修改的 rewardbench 脚本添加一个参数，num，为处理输出的条数，即只处理前 n 条数据，默认值 200，并加入到对应的 run_xxx.sh 中。
3. 在 scripts 下面现在有 self_synthesized 方法的代码，但是我们缺少不用 evaluation plan 的对照组，需要在 scripts 下面新建 baseline 文件夹，用与 self_synthesized 相同的方法评估数据（如果用到了完全一样的函数甚至可以直接从 self_synthesized 导入，prompt 使用：（原版 rewardbench 的 prompt）

   Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better.

   [[User Question]]
   {instruction}

   [The Start of Assistant A’s Answer]
   {response A}
   [The End of Assistant A’s Answer]

   [The Start of Assistant B’s Answer]
   {response B}
   [The End of Assistant B’s Answer]

4. 上述 prompt 删掉了现有代码平局的可能性，所有评分代码中相应的 0.5 分也得删掉。
5. scripts/self_synthesized/evaluate 文件夹下的代码可以直接复制一份到 baseline 文件夹下，预期 baseline 的 judge 输出数据格式和 self_synthesized 一样的话，可以直接用。万一不一样的话需要修改。

你仅按上述要求完成所有的代码工作，结束后不要运行这些代码，我会完整审核修改后自己运行。
