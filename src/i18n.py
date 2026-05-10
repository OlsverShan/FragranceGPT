"""
i18n: English / 中文 language support for FragranceGPT Streamlit app.
Usage:
    from src.i18n import t, set_lang, get_lang, LANGUAGES
    t("title")  ->  "🌸 FragranceGPT" or "🌸 香水智选" depending on lang
"""
import streamlit as st

LANGUAGES = {
    "en": "English",
    "zh": "中文",
}

# ── Accord name translations (English → 中文) ──
ACCORD_ZH = {
    "alcohol": "酒香",
    "aldehydic": "醛香",
    "almond": "杏仁",
    "amber": "琥珀",
    "animalic": "动物香",
    "anis": "茴香",
    "aquatic": "水生调",
    "aromatic": "芳香调",
    "asphault": "沥青",
    "balsamic": "香脂",
    "beeswax": "蜂蜡",
    "bitter": "苦味",
    "brown scotch tape": "胶带",
    "cacao": "可可",
    "camphor": "樟脑",
    "cannabis": "大麻",
    "caramel": "焦糖",
    "champagne": "香槟",
    "cherry": "樱桃",
    "chocolate": "巧克力",
    "cinnamon": "肉桂",
    "citrus": "柑橘调",
    "clay": "黏土",
    "coca-cola": "可乐",
    "coconut": "椰子",
    "coffee": "咖啡",
    "conifer": "针叶林",
    "creamy": "奶油",
    "earthy": "泥土",
    "floral": "花香调",
    "fresh": "清新调",
    "fresh spicy": "清新辛香",
    "fruity": "果香调",
    "gasoline": "汽油",
    "gourmand": "美食调",
    "green": "绿意调",
    "herbal": "草本",
    "honey": "蜂蜜",
    "hot iron": "热铁",
    "industrial glue": "工业胶水",
    "iris": "鸢尾",
    "lactonic": "奶香",
    "lavender": "薰衣草",
    "leather": "皮革",
    "marine": "海洋调",
    "metallic": "金属",
    "mineral": "矿物",
    "mossy": "苔藓",
    "musky": "麝香",
    "nutty": "坚果",
    "oily": "油脂",
    "oriental": "东方调",
    "oud": "乌木",
    "ozonic": "臭氧",
    "paper": "纸张",
    "patchouli": "广藿香",
    "plastic": "塑料",
    "powdery": "粉香",
    "rose": "玫瑰",
    "rubber": "橡胶",
    "rum": "朗姆酒",
    "salty": "咸味",
    "sand": "沙粒",
    "savory": "咸香",
    "smoky": "烟熏",
    "soapy": "皂感",
    "soft spicy": "柔和辛香",
    "sour": "酸味",
    "spicy": "辛香调",
    "sweet": "甜味",
    "terpenic": "萜烯",
    "tobacco": "烟草",
    "tropical": "热带果香",
    "tuberose": "晚香玉",
    "vanilla": "香草",
    "vinyl": "乙烯基",
    "violet": "紫罗兰",
    "vodka": "伏特加",
    "warm spicy": "温暖辛香",
    "whiskey": "威士忌",
    "white floral": "白花香调",
    "wine": "红酒",
    "woody": "木质调",
    "yellow floral": "黄花香调",
}

# ── Note/ingredient name translations (English → 中文) ──
# Covers top ~200 most frequent notes in the dataset
NOTE_ZH = {
    # Top 30
    "musk": "麝香", "bergamot": "佛手柑", "sandalwood": "檀木", "jasmine": "茉莉",
    "amber": "琥珀", "patchouli": "广藿香", "vanilla": "香草", "rose": "玫瑰",
    "cedar": "雪松", "mandarin orange": "橘子", "vetiver": "香根草", "tonka bean": "零陵香豆",
    "lemon": "柠檬", "lavender": "薰衣草", "orange blossom": "橙花",
    "lily-of-the-valley": "铃兰", "pink pepper": "粉红胡椒", "cardamom": "小豆蔻",
    "violet": "紫罗兰", "iris": "鸢尾", "grapefruit": "西柚", "ylang-ylang": "依兰",
    "leather": "皮革", "geranium": "天竺葵", "oakmoss": "橡苔", "peach": "桃子",
    "freesia": "小苍兰", "white musk": "白麝香", "benzoin": "安息香", "cinnamon": "肉桂",
    "orange": "橙子", "neroli": "橙花油", "black currant": "黑醋栗", "peony": "牡丹",
    "nutmeg": "肉豆蔻", "agarwood (oud)": "沉香(乌木)", "ginger": "生姜", "pear": "梨子",
    "tuberose": "晚香玉", "incense": "焚香", "saffron": "藏红花", "labdanum": "劳丹脂",
    "raspberry": "覆盆子", "woody notes": "木质香", "magnolia": "木兰", "heliotrope": "天芥菜",
    "ambergris": "龙涎香", "pepper": "胡椒", "gardenia": "栀子花", "apple": "苹果",
    "mint": "薄荷", "coriander": "芫荽", "green notes": "绿意", "black pepper": "黑胡椒",
    "virginia cedar": "弗吉尼亚雪松", "citruses": "柑橘", "plum": "李子",
    "woodsy notes": "木质香", "guaiac wood": "愈创木", "violet leaf": "紫罗兰叶",
    "pineapple": "菠萝", "carnation": "康乃馨", "sage": "鼠尾草", "moss": "苔藓",
    "lime": "青柠", "orchid": "兰花", "vanille": "香草", "jasmine sambac": "沙巴茉莉",
    "lily": "百合", "aldehydes": "醛香", "basil": "罗勒", "caramel": "焦糖",
    "tobacco": "烟草", "cashmere wood": "羊绒木", "tangerine": "红橘", "rosemary": "迷迭香",
    "osmanthus": "桂花", "galbanum": "白松香", "honey": "蜂蜜", "orris root": "鸢尾根",
    "petitgrain": "苦橙叶", "olibanum": "乳香", "coconut": "椰子", "clary sage": "快乐鼠尾草",
    "cloves": "丁香", "cypress": "柏木", "cassis": "黑醋栗芽", "amalfi lemon": "阿马尔菲柠檬",
    "litchi": "荔枝", "ambroxan": "降龙涎醚", "mimosa": "金合欢", "cashmeran": "开司米酮",
    "bulgarian rose": "保加利亚玫瑰", "almond": "杏仁", "artemisia": "艾蒿", "lotus": "莲花",
    "apricot": "杏子", "floral notes": "花香", "cyclamen": "仙客来", "sea notes": "海洋香",
    "spices": "辛香料", "praline": "果仁糖", "bitter orange": "苦橙", "orris": "鸢尾",
    "myrrh": "没药", "coffee": "咖啡", "red berries": "红浆果",
    "ambrette (musk mallow)": "黄葵籽", "melon": "蜜瓜", "juniper": "杜松",
    "honeysuckle": "金银花", "turkish rose": "土耳其玫瑰", "juniper berries": "杜松子",
    "water lily": "睡莲", "suede": "麂皮", "tea": "茶",
    "african orange flower": "非洲橙花", "narcissus": "水仙", "damask rose": "大马士革玫瑰",
    "green apple": "青苹果", "white flowers": "白花", "rum": "朗姆酒",
    "fruity notes": "果香", "styrax": "苏合香", "red apple": "红苹果",
    "frangipani": "鸡蛋花", "passionfruit": "百香果",
    "cypriol oil or nagarmotha": "莎草油", "elemi": "榄香脂", "amberwood": "琥珀木",
    "blackberry": "黑莓", "madagascar vanilla": "马达加斯加香草", "lilac": "丁香花",
    "thyme": "百里香", "clove": "丁香", "anise": "茴芹", "oak moss": "橡苔",
    "fig": "无花果", "birch": "桦木", "rhubarb": "大黄", "cacao": "可可",
    "hyacinth": "风信子", "water notes": "水漾", "yuzu": "柚子", "star anise": "八角",
    "blood orange": "血橙", "caraway": "葛缕子", "haitian vetiver": "海地香根草",
    "civet": "灵猫香", "mango": "芒果", "strawberry": "草莓",
    "brazilian rosewood": "巴西玫瑰木", "green leaves": "绿叶",
    "tiare flower": "提亚蕾花", "cherry": "樱桃", "calabrian bergamot": "卡拉布里亚佛手柑",
    "green tea": "绿茶", "spicy notes": "辛香", "opoponax": "红没药",
    "palisander rosewood": "紫檀木", "angelica": "当归", "cumin": "孜然",
    "watermelon": "西瓜", "watery notes": "水感", "licorice": "甘草",
    "sicilian lemon": "西西里柠檬", "pomegranate": "石榴", "milk": "牛奶",
    "sugar": "糖", "castoreum": "海狸香", "tolu balsam": "吐鲁香脂",
    "pink grapefruit": "粉红西柚", "egyptian jasmine": "埃及茉莉",
    "bourbon vanilla": "波旁香草", "white woods": "白木", "tarragon": "龙蒿",
    "myrhh": "没药", "sandalowood": "檀木", "may rose": "五月玫瑰",
    "white amber": "白琥珀", "red currant": "红醋栗", "peru balsam": "秘鲁香脂",
    "powdery notes": "粉感", "atlas cedar": "大西洋雪松", "clementine": "克莱门氏小柑橘",
    "immortelle": "蜡菊", "citron": "枸橼", "chamomile": "洋甘菊", "mate": "马黛茶",
    "papyrus": "纸莎草", "french labdanum": "法国劳丹脂", "oak": "橡木",
    "coumarin": "香豆素", "fig leaf": "无花果叶", "hazelnut": "榛子",
    "lemon verbena": "柠檬马鞭草", "bamboo": "竹子", "salt": "海盐",
    "nectarine": "油桃", "pomelo": "柚子",
    # More common notes
    "cedarwood": "雪松木", "sage oil": "鼠尾草精油", "black tea": "红茶",
    "white tea": "白茶", "tobacco leaf": "烟草叶", "tonka": "零陵香豆",
    "sweet notes": "甜香", "earthy notes": "泥土香", "fresh notes": "清新",
    "aromatic notes": "芳香", "balsamic notes": "香脂", "citrus notes": "柑橘",
    "smoky notes": "烟熏", "marine notes": "海洋", "ozonic notes": "臭氧",
    "herbal notes": "草本", "oriental notes": "东方调", "gourmand notes": "美食调",
    "animalic notes": "动物香", "soapy notes": "皂感", "creamy notes": "奶香",
    "warm spicy notes": "温暖辛香", "soft spicy notes": "柔和辛香",
    "fresh spicy notes": "清新辛香", "floral woody notes": "花香木质",
    "musk notes": "麝香", "amber notes": "琥珀香",
    # Flowers
    "rose de mai": "五月玫瑰", "taif rose": "塔伊夫玫瑰", "centifolia rose": "百叶玫瑰",
    "white rose": "白玫瑰", "wild rose": "野玫瑰", "rose absolute": "玫瑰净油",
    "rose oil": "玫瑰精油", "grasse rose": "格拉斯玫瑰", "rose water": "玫瑰花水",
    # Woods
    "sandalwood oil": "檀木精油", "mysore sandalwood": "迈索尔檀木",
    "australian sandalwood": "澳洲檀木", "indian sandalwood": "印度檀木",
    "white sandalwood": "白檀木", "cedar wood": "雪松木",
    # Citrus
    "sicilian bergamot": "西西里佛手柑", "italian bergamot": "意大利佛手柑",
    "calabrian lemon": "卡拉布里亚柠檬", "sicilian orange": "西西里橙子",
    "italian lemon": "意大利柠檬", "mediterranean citrus": "地中海柑橘",
    # Spices
    "white pepper": "白胡椒", "pink peppercorn": "粉红胡椒粒",
    "sichuan pepper": "花椒", "allspice": "多香果",
    # Fruits
    "coconut water": "椰子水", "coconut milk": "椰奶",
    "dried fruits": "干果", "tropical fruits": "热带水果",
    "citrus fruits": "柑橘类水果", "red fruits": "红色水果",
    # Others
    "vanilla absolute": "香草净油", "vanilla bean": "香草荚",
    "tonka bean absolute": "零陵香豆净油", "benzoin resin": "安息香树脂",
    "frankincense": "乳香", "elemi resin": "榄香树脂",
    "iris butter": "鸢尾浸膏", "iris root": "鸢尾根",
    "violet flower": "紫罗兰花", "violet petals": "紫罗兰花瓣",
    "lily of the valley": "铃兰", "tuberose absolute": "晚香玉净油",
    "jasmine absolute": "茉莉净油", "jasmine tea": "茉莉花茶",
    "neroli oil": "橙花油", "neroli blossom": "橙花",
    "lavender oil": "薰衣草精油", "lavender absolute": "薰衣草净油",
    "patchouli oil": "广藿香精油", "patchouli leaf": "广藿香叶",
    "vetiver oil": "香根草精油", "vetiver root": "香根草根",
    "oakmoss absolute": "橡苔净油", "tree moss": "树苔",
    "seaweed": "海藻", "algae": "藻类", "driftwood": "浮木",
    "mineral notes": "矿物", "metallic notes": "金属感",
    "aldehydic notes": "醛感", "powder": "粉末",
    "cotton candy": "棉花糖", "chocolate": "巧克力",
    "dark chocolate": "黑巧克力", "white chocolate": "白巧克力",
    "caramel notes": "焦糖", "honey notes": "蜂蜜",
    "whiskey": "威士忌", "vodka": "伏特加", "cognac": "干邑",
    "champagne": "香槟", "red wine": "红酒", "white wine": "白酒",
    "gin": "金酒", "absinthe": "苦艾酒",
    "smoke": "烟", "incense notes": "焚香", "ash": "灰烬",
    "leather notes": "皮革", "suede notes": "麂皮",
    "wax": "蜡", "beeswax": "蜂蜡", "honeycomb": "蜂巢",
    "rice": "大米", "sesame": "芝麻", "almond milk": "杏仁奶",
    "pistachio": "开心果", "walnut": "核桃", "chestnut": "栗子",
    "mushroom": "蘑菇", "truffle": "松露", "earth": "泥土",
    "rain": "雨水", "ozone": "臭氧", "air": "空气",
    # Common descriptors
    "dry": "干", "fresh": "清新", "sweet": "甜", "warm": "温暖",
    "cool": "清凉", "soft": "柔和", "rich": "浓郁", "light": "轻盈",
    "deep": "深沉", "dark": "暗黑", "bright": "明亮", "clean": "洁净",
    "dirty": "脏感", "green": "绿意", "creamy": "奶油", "buttery": "黄油",
    "milky": "奶感", "nutty": "坚果", "spicy": "辛香", "smoky": "烟熏",
    "earthy": "泥土", "woody": "木质", "floral": "花香", "fruity": "果香",
    "powdery": "粉感", "soapy": "皂感", "metallic": "金属", "salty": "咸",
    "sour": "酸", "bitter": "苦", "savory": "咸香",
    # Fixes for clean column names
    "fruity notes": "果香",
    "green notes": "绿意",
    "floral notes": "花香",
    "woody notes": "木质香",
    "spicy notes": "辛香",
    "citrus notes": "柑橘",
    "musky notes": "麝香",
    "amber notes": "琥珀香",
    "powdery notes": "粉感",
    "sweet notes": "甜香",
    "fresh notes": "清新",
    "earthy notes": "泥土香",
    "smoky notes": "烟熏",
    "aromatic notes": "芳香",
    "balsamic notes": "香脂",
    "marine notes": "海洋",
    "ozonic notes": "臭氧",
    "herbal notes": "草本",
    "oriental notes": "东方调",
    "gourmand notes": "美食调",
    "animalic notes": "动物香",
    "soapy notes": "皂感",
    "creamy notes": "奶香",
    "warm spicy notes": "温暖辛香",
    "soft spicy notes": "柔和辛香",
    "fresh spicy notes": "清新辛香",
    "leather notes": "皮革香",
    "suede notes": "麂皮香",
    "mineral notes": "矿物香",
    "metallic notes": "金属香",
    "aldehydic notes": "醛感",
    "incense notes": "焚香",
    "water notes": "水漾",
    "sea notes": "海洋香",
    "aquatic notes": "水生",
    "white floral notes": "白花香",
    "yellow floral notes": "黄花香",
    "tropical notes": "热带风情",
    "wine notes": "酒香",
    "coffee notes": "咖啡香",
    "tea notes": "茶香",
    "tobacco notes": "烟草香",
    "cacao notes": "可可香",
    "vanilla notes": "香草香",
    "caramel notes": "焦糖香",
    "honey notes": "蜂蜜香",
    "nutty notes": "坚果香",
    "milky notes": "奶香",
    "lactonic notes": "奶感",
}

TRANSLATIONS = {
    # ── Page config ──
    "page_title": {"en": "FragranceGPT", "zh": "香水智选"},
    "page_icon":  {"en": "🌸", "zh": "🌸"},

    # ── Title ──
    "title": {"en": "🌸 FragranceGPT", "zh": "🌸 香水智选"},
    "caption": {
        "en": "Perfume Recommendation & Formula Rating — Powered by RAG + DeepSeek",
        "zh": "香水推荐与配方评分 — RAG + DeepSeek 驱动",
    },

    # ── Tabs ──
    "tab_recommend":  {"en": "🎯 Perfume Recommendation", "zh": "🎯 香水推荐"},
    "tab_rating":     {"en": "📊 Rating Predictor", "zh": "📊 评分预测"},
    "tab_analytics":  {"en": "📈 Market Analytics", "zh": "📈 市场分析"},

    # ── Tab 1: Recommendation ──
    "rec_subtitle": {"en": "Find High-Rated Perfumes Matching Your Taste", "zh": "发现符合你口味的高评分香水"},
    "rec_pick_accords": {
        "en": "Pick accords you like (e.g., citrus, woody, floral):",
        "zh": "选择你喜欢的香调（例如：柑橘、木质、花香）：",
    },
    "rec_placeholder": {"en": "Choose 1-5 accords...", "zh": "选择 1-5 个香调..."},
    "rec_help": {
        "en": "Select 1 or more accords. More accords = more precise recommendations.",
        "zh": "选择至少1个香调，越多越精准。",
    },
    "rec_button": {"en": "🔍 Get Recommendations", "zh": "🔍 获取推荐"},
    "rec_no_api": {"en": "API key not configured. Showing vector search results only.", "zh": "API 密钥未配置，仅展示向量检索结果。"},
    "rec_predicted_formula": {"en": "🧪 AI-Predicted Fragrance Formula", "zh": "🧪 AI 预测香水配方"},
    "rec_top_notes":    {"en": "🍋 Top Notes", "zh": "🍋 前调"},
    "rec_middle_notes": {"en": "🌸 Middle Notes", "zh": "🌸 中调"},
    "rec_base_notes":   {"en": "🪵 Base Notes", "zh": "🪵 后调"},
    "rec_xgb_rating":   {"en": "XGBoost rating", "zh": "XGBoost 评分"},
    "rec_expert_panel": {"en": "👥 Get Expert Opinions (8-Persona Panel)", "zh": "👥 获取专家意见（八人评审团）"},
    "rec_rate_experts": {"en": "Rate with 8 Experts", "zh": "八位专家联合评审"},
    "rec_evaluating":   {"en": "8 personas evaluating from conflicting perspectives...", "zh": "八位专家从不同视角评审中..."},
    "rec_weighted_score": {"en": "Weighted Overall Score", "zh": "加权综合评分"},
    "rec_polarization":   {"en": "Polarization", "zh": "分歧度"},
    "rec_range":          {"en": "range", "zh": "极差"},
    "rec_top5_title":     {"en": "🏆 Top-5 High-Rated Perfumes You Might Love", "zh": "🏆 你可能喜欢的 Top-5 高分香水"},
    "rec_match_score":    {"en": "Match Score", "zh": "匹配分数"},
    "rec_match_help":     {"en": "Composite: similarity + content + quality + diversity", "zh": "综合：相似度 + 内容 + 品质 + 多样性"},
    "rec_note_overlap":   {"en": "Note Overlap", "zh": "香调重叠度"},
    "rec_no_selection":   {"en": "Please select at least 1 accord.", "zh": "请至少选择一个香调。"},

    # ── Tab 2: Rating Predictor ──
    "rate_subtitle":     {"en": "Rate a Formula: XGBoost + Multi-Persona LLM", "zh": "评分预测：XGBoost + 多人设 LLM"},
    "rate_no_model":     {"en": "Run `python train_rating_predictor.py` first.", "zh": "请先运行 `python train_rating_predictor.py` 训练模型。"},
    "rate_enter_notes":  {"en": "Enter a perfume formula to estimate its Rating Value.", "zh": "输入香水配方，预估评分。"},
    "rate_top_input":    {"en": "Top Notes", "zh": "前调"},
    "rate_mid_input":    {"en": "Middle Notes", "zh": "中调"},
    "rate_base_input":   {"en": "Base Notes", "zh": "后调"},
    "rate_top_ph":       {"en": "bergamot, lemon, neroli", "zh": "佛手柑, 柠檬, 橙花"},
    "rate_mid_ph":       {"en": "jasmine, rose, geranium", "zh": "茉莉, 玫瑰, 天竺葵"},
    "rate_base_ph":      {"en": "sandalwood, vanilla, musk", "zh": "檀木, 香草, 麝香"},
    "rate_xgb_btn":      {"en": "⚡ XGBoost (Instant)", "zh": "⚡ XGBoost（即时）"},
    "rate_llm_btn":      {"en": "👥 Multi-Persona LLM (Deep)", "zh": "👥 多人设 LLM（深度）"},
    "rate_no_notes":     {"en": "Please enter at least a few notes.", "zh": "请输入至少几个香调。"},
    "rate_xgb_title":    {"en": "XGBoost Prediction", "zh": "XGBoost 预测"},
    "rate_quality_tier": {"en": "Quality Tier", "zh": "品质等级"},
    "rate_xgb_note1":    {"en": "Held-out MAE: 0.17 | Pearson r: 0.53 | R^2: 0.28", "zh": "测试集 MAE: 0.17 | Pearson r: 0.53 | R²: 0.28"},
    "rate_xgb_note2":    {"en": "Model relies 82% on note text patterns, 12% on accords.", "zh": "模型 82% 依赖香调文本，12% 依赖香型分类。"},
    "rate_llm_weighted": {"en": "Weighted Overall", "zh": "加权综合"},
    "rate_result_note":  {"en": "highest: {}, lowest: {}", "zh": "最高: {}，最低: {}"},

    # ── Tab 2: Fused Rating ──
    "fuse_btn":          {"en": "🤝 Fused Rating (XGBoost + Experts)", "zh": "🤝 融合评分（XGBoost + 专家）"},
    "fuse_title":        {"en": "Fused Rating", "zh": "融合评分"},
    "fuse_breakdown":    {"en": "Score Breakdown", "zh": "评分拆解"},
    "fuse_xgb_base":     {"en": "XGBoost Baseline", "zh": "XGBoost 基线"},
    "fuse_persona_raw":  {"en": "8-Persona Raw", "zh": "八人原始评分"},
    "fuse_persona_cal":  {"en": "8-Persona Calibrated", "zh": "八人校准评分"},
    "fuse_adjustment":   {"en": "Persona Adjustment", "zh": "专家调整量"},
    "fuse_confidence":   {"en": "Fusion Confidence", "zh": "融合置信度"},
    "fuse_agreement":    {"en": "Expert Agreement", "zh": "专家一致性"},
    "fuse_no_persona":   {"en": "XGBoost only — persona panel unavailable", "zh": "仅 XGBoost — 专家团不可用"},
    "fuse_persona_only": {"en": "8-Persona only — XGBoost model not trained", "zh": "仅专家团 — XGBoost 模型未训练"},

    # ── Tab 3: Market Analytics ──
    "analytics_subtitle": {"en": "Fragrance Market Insights", "zh": "香水市场洞察"},
    "analytics_accord_rank": {"en": "Accord Rankings", "zh": "香调排名"},
    "analytics_brand_rank":  {"en": "Brand Rankings", "zh": "品牌排名"},
    "analytics_rating_dist": {"en": "Rating Distribution", "zh": "评分分布"},

    # ── Analytics: Accord tab ──
    "ana_top_combos":  {"en": "Top Accord Combinations", "zh": "最佳香调组合"},
    "ana_combo_chart":  {"en": "Highest-Rated Accord Combinations", "zh": "最高评分香调组合"},
    "ana_avg_score":    {"en": "Avg Bayesian Score", "zh": "平均贝叶斯评分"},
    "ana_accord_corr":  {"en": "Accord-Rating Correlation", "zh": "香调-评分相关性"},
    "ana_corr_chart":   {"en": "Accords Most Associated with High Ratings", "zh": "最关联高评分的香调"},
    "ana_deviation":    {"en": "Deviation from Global Mean", "zh": "偏离全局均值"},
    "ana_common_notes": {"en": "Most Common Notes in High-Rated Perfumes (≥4.0)", "zh": "高分香水中最常见香调（≥4.0）"},
    "ana_note_freq":    {"en": "Note Frequency in Top-Rated Perfumes", "zh": "高分香水中的香调频率"},
    "ana_pct":          {"en": "% of Perfumes", "zh": "出现比例"},

    # ── Analytics: Brand tab ──
    "ana_top_brands": {"en": "Top Brands by Bayesian Score", "zh": "品牌贝叶斯评分排名"},
    "ana_brand_col":  {"en": "Brand", "zh": "品牌"},
    "ana_avg_score_col": {"en": "Avg Score", "zh": "平均分"},
    "ana_perfumes_col":  {"en": "Perfumes", "zh": "香水数"},

    # ── Analytics: Rating Distribution tab ──
    "ana_rating_dist_chart": {"en": "Perfume Rating Distribution (Rating Count > 50)", "zh": "香水评分分布（评论数 > 50）"},
    "ana_rating_range_x":    {"en": "Rating Range", "zh": "评分区间"},
    "ana_count_y":           {"en": "Number of Perfumes", "zh": "香水数量"},
    "ana_total_perfumes":    {"en": "Total Perfumes", "zh": "总香水数"},
    "ana_high_conf":         {"en": "With Rating Count > 50", "zh": "高置信度（评论 > 50）"},
    "ana_global_mean":       {"en": "Global Mean Rating", "zh": "全局平均评分"},
    "ana_rating_range":      {"en": "Rating Range", "zh": "评分范围"},

    # ── Multi-persona polarization labels ──
    "polarization_consensus":      {"en": "consensus", "zh": "一致认可"},
    "polarization_moderate":       {"en": "moderate spread", "zh": "略有分歧"},
    "polarization_divided":        {"en": "divided opinions", "zh": "意见分化"},
    "polarization_highly":         {"en": "highly polarizing", "zh": "极具争议"},

    # ── Quality tiers ──
    "tier_masterpiece":   {"en": "Masterpiece", "zh": "神作"},
    "tier_excellent":     {"en": "Excellent", "zh": "优秀"},
    "tier_good":          {"en": "Good", "zh": "不错"},
    "tier_average":       {"en": "Average", "zh": "普通"},
    "tier_below_avg":     {"en": "Below Average", "zh": "较差"},
    "tier_poor":          {"en": "Poor", "zh": "糟糕"},

    # ── Language selector ──
    "lang_label": {"en": "Language / 语言", "zh": "Language / 语言"},

    # ── Purchase Links ──
    "purchase_btn": {"en": "🛒 Find Purchase Links", "zh": "🛒 查找购买链接"},
    "purchase_title": {"en": "🛒 Where to Buy: {}", "zh": "🛒 购买渠道：{}"},
    "purchase_loading": {"en": "Searching for purchase links...", "zh": "正在搜索购买链接..."},
    "purchase_price": {"en": "Price", "zh": "价格区间"},
    "purchase_retailer": {"en": "Retailer", "zh": "零售商"},
    "purchase_similar": {"en": "🔄 Similar Perfumes You Might Like", "zh": "🔄 你可能喜欢的类似香水"},
    "purchase_similar_why": {"en": "Why similar", "zh": "相似原因"},
    "purchase_no_api": {"en": "API key not configured — cannot search for purchase links.", "zh": "API 密钥未配置，无法搜索购买链接。"},
    "purchase_error": {"en": "Search failed: {}", "zh": "搜索失败：{}"},
    "purchase_no_links": {"en": "No purchase links found for this perfume.", "zh": "未找到此香水的购买链接。"},
    "purchase_disclaimer": {"en": "Links generated from AI knowledge. Please verify availability and pricing before purchasing.", "zh": "链接由 AI 知识生成，购买前请核实库存和价格。"},

    # ── Sales Expert (Scent Concierge) ──
    "sales_title":          {"en": "🤵 Perfume Advisor — Tell Me What You Need", "zh": "🤵 香水顾问 — 告诉我你的需求"},
    "sales_subtitle":       {"en": "Not sure what accords you like? Describe your ideal fragrance scenario, and our AI sales expert will analyze your needs and recommend the perfect scent direction.", "zh": "不确定你喜欢什么香调？描述你理想的用香场景，让 AI 香水顾问分析你的需求，推荐最适合的香水方向。"},
    "sales_desc_label":     {"en": "Describe your ideal fragrance", "zh": "描述你心目中的香水"},
    "sales_desc_ph":        {"en": "e.g. I want something fresh and clean for everyday office wear, not too strong but still noticeable when someone gets close...", "zh": "例如：我想要一款清新干净的日常通勤香，不要太浓，但靠近时能让人感受到..."},
    "sales_scenario_label": {"en": "Usage scenario (optional)", "zh": "使用场景（选填）"},
    "sales_scenario_ph":    {"en": "e.g. office, date night, weekend casual, wedding...", "zh": "例如：办公室、约会、周末休闲、婚礼..."},
    "sales_target_label":   {"en": "Target audience / mood (optional)", "zh": "面向对象 / 心情（选填）"},
    "sales_target_ph":      {"en": "e.g. for myself, to impress someone, to feel confident...", "zh": "例如：为自己、吸引他人、提升自信..."},
    "sales_btn":            {"en": "🎯 Analyze My Needs & Recommend", "zh": "🎯 分析需求并推荐"},
    "sales_analyzing":      {"en": "Our fragrance expert is analyzing your needs...", "zh": "香水顾问正在分析你的需求..."},
    "sales_needs_title":    {"en": "🧠 Needs Analysis", "zh": "🧠 需求分析"},
    "sales_accords_title":  {"en": "🎨 Recommended Accords", "zh": "🎨 推荐香调方向"},
    "sales_blend_title":    {"en": "✨ AI Blend Direction", "zh": "✨ AI 调香方向"},
    "sales_top5_title":     {"en": "🏆 Top 5 Perfumes You Should Try", "zh": "🏆 你应该尝试的 5 款香水"},
    "sales_style":          {"en": "Style", "zh": "风格"},
    "sales_key_notes":      {"en": "Key Notes", "zh": "核心香调"},
    "sales_why_fits":       {"en": "Why It Fits You", "zh": "为何适合你"},
    "sales_error":          {"en": "Analysis failed: {}", "zh": "分析失败：{}"},

    # ── Misc ──
    "error": {"en": "Error", "zh": "错误"},
    "loading": {"en": "AI is crafting your fragrance profile...", "zh": "AI 正在为你调配香水配方..."},

    # ── History Tab ──
    "tab_history":          {"en": "🕐 History", "zh": "🕐 历史记录"},
    "history_title":        {"en": "Past Results", "zh": "历史结果"},
    "history_clear":        {"en": "🗑️ Clear All", "zh": "🗑️ 清空全部"},
    "history_empty":        {"en": "No history yet. Run a recommendation or rating to see it here.", "zh": "暂无历史记录。运行推荐或评分后在此查看。"},
    "history_filter":       {"en": "Filter by type", "zh": "按类型筛选"},
    "history_rerun":        {"en": "🔄 Re-run", "zh": "🔄 重新执行"},
    "history_delete":       {"en": "Delete", "zh": "删除"},
    "history_input":        {"en": "Input", "zh": "输入"},
    "history_result":       {"en": "Result", "zh": "结果"},
    "history_type_sales":   {"en": "🤵 Scent Concierge", "zh": "🤵 香水顾问"},
    "history_type_accord":  {"en": "🎯 Accord Selection", "zh": "🎯 香调推荐"},
    "history_type_persona": {"en": "👥 Expert Panel", "zh": "👥 专家评审"},
    "history_type_fused":   {"en": "🤝 Fused Rating", "zh": "🤝 融合评分"},
    "history_type_purchase":{"en": "🛒 Purchase Search", "zh": "🛒 购买查询"},
}


def get_lang():
    """Get current language from session state, default to English."""
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    return st.session_state.lang


def set_lang(lang):
    """Set language in session state."""
    st.session_state.lang = lang


def t(key, **kwargs):
    """Translate a key to the current language. Supports .format(**kwargs)."""
    lang = get_lang()
    text = TRANSLATIONS.get(key, {}).get(lang, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def accord_name(english_name):
    """Translate an accord name to current language."""
    lang = get_lang()
    if lang == "zh":
        return ACCORD_ZH.get(english_name.lower(), english_name)
    return english_name


def accord_names_bilingual(english_names):
    """Return accord names in current language, with bilingual format for Chinese.
    In EN mode: just the English name.
    In ZH mode: '中文 (English)' — helps users match both languages.
    """
    lang = get_lang()
    result = []
    for name in english_names:
        zh = ACCORD_ZH.get(name.lower(), name)
        if lang == "zh" and zh != name:
            result.append(f"{zh} ({name})")
        else:
            result.append(name)
    return result


def note_name(english_name):
    """Translate a single note/ingredient name to current language."""
    lang = get_lang()
    if lang == "zh":
        return NOTE_ZH.get(english_name.lower(), english_name)
    return english_name


def note_names_translated(english_names):
    """Translate a list of note names to current language.
    In EN mode: just English. In ZH mode: translated with English in parens.
    """
    lang = get_lang()
    result = []
    for name in english_names:
        name_clean = name.strip()
        zh = NOTE_ZH.get(name_clean.lower(), name_clean)
        if lang == "zh" and zh != name_clean:
            result.append(f"{zh} ({name_clean})")
        else:
            result.append(name_clean)
    return result


def quality_tier_i18n(rating):
    """Return (tier_key, color, icon) calibrated to Fragrantica distribution."""
    if rating >= 4.3:
        return "tier_masterpiece", "#B8860B", "🏆"
    elif rating >= 4.0:
        return "tier_excellent", "#2E7D32", "⭐"
    elif rating >= 3.5:
        return "tier_good", "#4CAF50", "👍"
    elif rating >= 3.0:
        return "tier_average", "#FF9800", "👌"
    elif rating >= 2.5:
        return "tier_below_avg", "#F44336", "👎"
    else:
        return "tier_poor", "#9E9E9E", "💀"


def stars_html_i18n(rating, show_tier=True):
    """Render star rating as HTML with localized quality tier badge."""
    full = int(rating)
    half = 1 if rating - full >= 0.25 else 0
    s = "★" * full + "☆" * (5 - full)
    tier_key, tier_color, tier_icon = quality_tier_i18n(rating)
    tier_name = t(tier_key)
    tier_badge = (f' <span style="background:{tier_color};color:white;padding:2px 10px;'
                  f'border-radius:12px;font-size:0.75em;margin-left:6px">{tier_icon} {tier_name}</span>') if show_tier else ""
    return f'<span style="color:#FFD700;font-size:1.2em">{s}</span> {rating:.2f}{tier_badge}'


def polarization_i18n(label_en):
    """Translate polarization label from result dict."""
    mapping = {
        "consensus": "polarization_consensus",
        "moderate spread": "polarization_moderate",
        "divided opinions": "polarization_divided",
        "highly polarizing": "polarization_highly",
    }
    key = mapping.get(label_en, label_en)
    return t(key)
