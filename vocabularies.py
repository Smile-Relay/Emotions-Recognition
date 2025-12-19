ADJECTIVES = [
    "干净的",
    "吵闹的",
    "普通的",
    "忙碌的",
    "老旧的",
    "新买的",
    "冰凉的",
    "温热的",
    "随手的",
    "偏暗的",
    "结实的",
    "空着的",
    "慢慢的",
    "随意的",
    "熟悉的",
    "顺滑的"
]

NOUNS = [
    "桌子",
    "椅子",
    "手机",
    "钥匙",
    "杯子",
    "书包",
    "门",
    "窗户",
    "披萨",
    "面包",
    "巧克力",
    "水果盘",
    "蛋糕",
    "咖啡杯",
    "雨伞",
    "笔记本"
]

def get_by_hex(hex_id: str):
    adjective, noun = map(lambda h: int(h, 16), hex_id)
    return ADJECTIVES[adjective], NOUNS[noun]

if __name__ == "__main__":
    print(get_by_hex(input()))