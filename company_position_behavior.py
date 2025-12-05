import numpy as np
import pandas as pd

np.random.seed(42)

# 基础配置
n_companies = 500
jobs_per_company_min = 1
jobs_per_company_max = 5      # 每家企业的岗位数范围
base_lambda_exposure = 800    # 曝光强度基准

industries = ["互联网", "金融", "制造业", "教育", "咨询", "零售"]
funding_stages = ["未融资", "天使轮", "A轮", "B轮", "C轮及以上", "上市公司"]
cities = ["北京", "上海", "深圳", "杭州", "新加坡", "广州"]
job_functions = ["算法工程师", "数据分析", "后端开发", "前端开发", "产品经理", "运营"]

# ======================
# 1. 企业侧数据
# ======================

company_ids = np.arange(1, n_companies + 1)

# 企业基础画像
company_profile = pd.DataFrame({
    "company_id": company_ids,
    "industry": np.random.choice(industries, size=n_companies),
    "company_size": np.random.choice([50, 100, 200, 500, 1000, 3000], size=n_companies),
    "funding_stage": np.random.choice(funding_stages, size=n_companies),
    "brand_level": np.random.randint(1, 6, size=n_companies),
    "salary_level": np.random.randint(1, 6, size=n_companies),
    "intl_flag": np.random.binomial(1, 0.3, size=n_companies),
    "top10_talent_ratio": np.round(np.random.uniform(0.05, 0.5, size=n_companies), 3),
    "avg_apply_cnt": np.random.randint(50, 300, size=n_companies),
    "hire_rate": np.round(np.random.uniform(0.05, 0.3, size=n_companies), 3)
})

# 企业软指标
company_soft_metrics = pd.DataFrame({
    "company_id": company_ids,
    "overall_rating": np.round(np.random.uniform(3.0, 5.0, size=n_companies), 2),
    "work_life_balance_score": np.round(np.random.uniform(2.5, 5.0, size=n_companies), 2),
    "salary_competitiveness": np.round(np.random.uniform(2.5, 5.0, size=n_companies), 2),
    "resume_response_rate": np.round(np.random.uniform(0.2, 0.9, size=n_companies), 3),
    "resume_response_time": np.round(np.random.uniform(0.5, 7.0, size=n_companies), 2),
    "offer_conversion_rate": np.round(np.random.uniform(0.1, 0.6, size=n_companies), 3)
})


def min_max(s: pd.Series) -> pd.Series:
    """min-max 归一化"""
    return (s - s.min()) / (s.max() - s.min() + 1e-6)


# 软指标归一化并构造 talent_care_index
soft_norm = pd.DataFrame({
    "overall_rating_n": min_max(company_soft_metrics["overall_rating"]),
    "wlb_n": min_max(company_soft_metrics["work_life_balance_score"]),
    "salary_comp_n": min_max(company_soft_metrics["salary_competitiveness"]),
    "resp_rate_n": min_max(company_soft_metrics["resume_response_rate"]),
    # 响应时间越短越好，这里取反
    "resp_time_n": 1 - min_max(company_soft_metrics["resume_response_time"]),
    "offer_conv_n": min_max(company_soft_metrics["offer_conversion_rate"])
})

talent_care_index = (
    0.2 * soft_norm["overall_rating_n"]
    + 0.2 * soft_norm["wlb_n"]
    + 0.15 * soft_norm["salary_comp_n"]
    + 0.2 * soft_norm["resp_rate_n"]
    + 0.1 * soft_norm["resp_time_n"]
    + 0.15 * soft_norm["offer_conv_n"]
)

company_tier_features = pd.DataFrame({
    "company_id": company_ids,
    "talent_care_index": np.round(talent_care_index, 3)
})

# 按 brand_level + talent_care_index 做简单三档分层
tier_score = company_profile["brand_level"] + 2 * company_tier_features["talent_care_index"]
q1, q2 = tier_score.quantile([0.33, 0.66])


def assign_tier(x: float) -> str:
    if x <= q1:
        return "T3"
    elif x <= q2:
        return "T2"
    else:
        return "T1"


company_tier_features["enterprise_tier"] = tier_score.apply(assign_tier)

# ======================
# 2. 岗位侧数据
# ======================

job_rows = []
job_id = 1

# 为每个企业生成若干岗位
for cid in company_ids:
    n_jobs = np.random.randint(jobs_per_company_min, jobs_per_company_max + 1)
    for _ in range(n_jobs):
        func = np.random.choice(job_functions)
        # 热门岗位标记
        hot_function = 1 if func in ["算法工程师", "数据分析", "后端开发"] else 0
        level = np.random.choice(["初级", "中级", "高级"])
        base = np.random.randint(8, 40)
        salary_min_k = base
        salary_max_k = base + np.random.randint(2, 15)

        job_rows.append({
            "job_id": job_id,
            "company_id": cid,
            "job_type": "全职",
            "job_level": level,
            "job_function": func,
            "city": np.random.choice(cities),
            "salary_min_k": salary_min_k,
            "salary_max_k": salary_max_k,
            "target_top10": np.random.binomial(1, 0.4),
            "online_days": np.random.randint(7, 60),
            "urgency": np.random.choice([1, 2, 3]),
            "hot_function": hot_function
        })
        job_id += 1

job_profile = pd.DataFrame(job_rows)

# 合并企业信息，便于后续生成曝光 / 投递
job_merged = (
    job_profile
    .merge(company_profile, on="company_id", how="left")
    .merge(company_tier_features, on="company_id", how="left")
)

# 2.2 曝光 / 投递 / 录用 

# 吸引力得分 attract_score
eps = np.random.normal(0, 0.8, size=len(job_merged))
attract_score = (
    0.8 * job_merged["brand_level"]
    + 0.7 * job_merged["salary_level"]
    + 0.5 * job_merged["intl_flag"]
    + 0.9 * job_merged["target_top10"]
    + 0.6 * job_merged["hot_function"]
    + eps
)

# 查看概率 p_view
p_view = 1 / (1 + np.exp(-attract_score))

# 曝光量：Poisson 分布
lambda_exposure = base_lambda_exposure * p_view
lambda_exposure = lambda_exposure.clip(20, None)
impressions = np.random.poisson(lam=lambda_exposure)

# 查看次数：在曝光基础上按 p_view 抽样
views = np.random.binomial(impressions, p_view.clip(0.05, 0.95))

# 投递次数：查看后的投递概率设为 0.3 * p_view
p_apply = (p_view * 0.3).clip(0.02, 0.6)
applies = np.random.binomial(views, p_apply)

# 录用次数：按企业级 hire_rate 再抽样
hire_rate = job_merged["hire_rate"].fillna(0.1).values
hires = np.random.binomial(applies, hire_rate.clip(0.02, 0.8))

job_stats = pd.DataFrame({
    "job_id": job_profile["job_id"],
    "impression_cnt": impressions,
    "view_cnt": views,
    "apply_cnt": applies,
    "hire_cnt": hires
})

# 2.3 定价标签

top10_ratio = job_merged["top10_talent_ratio"].fillna(0.1).values
expected_applies = applies * top10_ratio
v = 100
expected_value = v * expected_applies
ROI_target = 3.0

brand_level = job_merged["brand_level"].values
price_label = expected_value / ROI_target * (0.8 + 0.1 * brand_level)
price_label = np.clip(price_label, 100, 5000)

job_pricing_label = pd.DataFrame({
    "job_id": job_profile["job_id"],
    "expected_high_quality_applies": expected_applies.round(2),
    "expected_value": expected_value.round(2),
    "price_label": price_label.round(2)
})

# ======================
# 3. 行为日志数据 job_apply_logs
# ======================

logs = []
log_id = 1

# 这里只对有投递的岗位，模拟 view -> apply -> offer/reject 三类事件
for _, r in job_stats.iterrows():
    jid = int(r["job_id"])
    n_apply = int(r["apply_cnt"])

    for i in range(n_apply):
        uid = f"U_{jid}_{i+1}"
        base_time = pd.Timestamp("2025-01-01") + pd.to_timedelta(
            np.random.randint(0, 60), unit="D"
        )

        # view 事件
        logs.append({
            "log_id": log_id,
            "job_id": jid,
            "user_id": uid,
            "event_time": base_time,
            "event_type": "view",
            "is_high_quality_user": np.random.binomial(1, 0.4),
            "apply_result": None
        })
        log_id += 1

        # apply 事件
        logs.append({
            "log_id": log_id,
            "job_id": jid,
            "user_id": uid,
            "event_time": base_time + pd.to_timedelta(np.random.randint(0, 3), unit="D"),
            "event_type": "apply",
            "is_high_quality_user": np.random.binomial(1, 0.4),
            "apply_result": None
        })
        log_id += 1

        # offer / reject 结果
        result = np.random.choice(["offer", "reject"], p=[0.2, 0.8])
        logs.append({
            "log_id": log_id,
            "job_id": jid,
            "user_id": uid,
            "event_time": base_time + pd.to_timedelta(np.random.randint(3, 15), unit="D"),
            "event_type": result,
            "is_high_quality_user": np.random.binomial(1, 0.4),
            "apply_result": result
        })
        log_id += 1

job_apply_logs = pd.DataFrame(logs)

print("company_profile:", company_profile.shape)
print("company_soft_metrics:", company_soft_metrics.shape)
print("company_tier_features:", company_tier_features.shape)
print("job_profile:", job_profile.shape)
print("job_stats:", job_stats.shape)
print("job_pricing_label:", job_pricing_label.shape)
print("job_apply_logs:", job_apply_logs.shape)

# 如需导入 MySQL / ClickHouse，可按需导出 CSV
company_profile.to_csv("company_profile.csv", index=False)
company_soft_metrics.to_csv("company_soft_metrics.csv", index=False)
company_tier_features.to_csv("company_tier_features.csv", index=False)
job_profile.to_csv("job_profile.csv", index=False)
job_stats.to_csv("job_stats.csv", index=False)
job_pricing_label.to_csv("job_pricing_label.csv", index=False)
job_apply_logs.to_csv("job_apply_logs.csv", index=False)

