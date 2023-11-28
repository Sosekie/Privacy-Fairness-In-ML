我们希望每次处理一个新的数据集的时候，使用同样的函数将其处理为各个变量：x-数据特征, y-未来结果, z-对决策带来bias的特征, a-过往决策。
在本次数据集中，y对应第30列”Expected Cumulative grade point average in the graduation“；z对应第2列”Sex“以及第31列”Course ID“；a是我们根据第29列”Cumulative grade point average in the last semester“以及第32列”OUTPUT Grade“生成的决策，1表示”被录取“，0表示“被拒绝”；x是数据中分出z和y之后剩下的其他所有特征。

这里，我们假设一个问题：根据学生的现有成绩（现有GPA以及各个课程的成绩）——x，以及预期的未来GPA——y，判断学校该不该给学生master offer——a。\
如果是按照individual academic performance也就是学生的现有GPA以及各个课程的成绩给offer，那么我们规定：只有现有GPA在4以上的，才能被考虑；同时录取名额有限制：只有m名同学可以被录取，那么我们再规定：当GPA相同时，课程成绩越高的学生会被录取。这样我们就得到了初始的决策结果——a_original。\
$\pi_0(a_t | x_t, z_t)$\
这个决策结果很显然是不合理的——因为不同专业的人的GPA由不同课程的老师给出，而各个老师给的成绩的分布是有差距的，有些老师给的分数相对较高，有些老师则较低；另外，当GPA相同时比较课程成绩也是不合理的，因为不同的课程难度并不相同，并不能用单一的评价标准进行衡量。所以接下来我们要对这个policy进行优化。\
对于Group Fairness，这里有几个需要平衡的问题：第一是性别——比如课本里提到的一些高中老师不鼓励女生学习数学；第二是专业——有些专业课程成绩的均值和中位数都明显大于其他专业的课程成绩，这会导致这些专业的GPA相对更高。这些都可以被划分为不同的群体——z，我们希望其满足以下公式也就是Equality of Opportunity：\
$P^{\pi}_{\theta}(a_t | z_t) = P^{\pi_{\theta}}(a_t)$.\
同时，对于Demographic Parity，我们希望其满足以下公式：\
$P^{\pi}_{\theta}(y_t | z_t) = P^{\pi_{\theta}}(y_t)$.\
For Calibration, 我们希望其满足以下公式：\
$P^{\pi}_{\theta}(y_t | a_t, z_t) = P^{\pi}_{\theta}(y_t | a_t)$.\
For Balance, 我们希望其满足以下公式：\
$P^{\pi}_{\theta}(a_t | y_t, z_t) = P^{\pi}_{\theta}(a_t | y_t)$.\
对于Utility function,因为y的值越大，说明被录取的学生预期取得的GPA越高，所以我们考虑以下函数：\
$U(a, y) = (a-\frac{y}{y_{max}})^2$\
这个函数说明当被录取的学生的预期GPA越高，函数取值越小，这种情况下我们希望U越小越好。\
对于Fairness function，因为我们知道严格的独立不相关是很难在实际中存在的，所以我们采用以下函数：\
$F(a, y) = (P^{\pi}_{\theta}(a_t | y_t, z_t) - P^{\pi}_{\theta}(a_t | y_t))^2$\
这种情况下我们希望F越小越好。注意，这里的$P^{\pi}_{\theta}(a_t | y_t, z_t)$只是一个示例，实际上我们会计算上面提到的四个独立不相关条件。\
然后我们就可以用Unconstrained optimisation以及设定的$\lambda$去构建目标函数以及最优化它：\
$\min_{\pi} (1 - \lambda)U(\pi, \theta) + \lambda F(\pi, \theta)$\
其中，对于x, y, z, a, 我们都不会做改变，我们最优化的目标只有新的policy函数中的可训练权重a和b：\
$\pi(a | x, z) = ax + bz$\
这样得到新的policy函数之后，我们就可以在下次招生的时候，对所有学生做更加fairness的预测。

对于Individual Fairness，我们之前提到了”当GPA相同时，课程成绩越高的学生会被录取。“是不合理的，所以我们希望充分考虑一个学生的所有方面，来使得决策更加平滑。