class AddNetworkEmbedding:
    """
    添加网络嵌入特征到样本字典。

    注意：当前训练脚本未使用该 transform，但保留原项目接口以保证可扩展性。
    """

    def __init__(self, emb_s, emb_t):
        self.emb_s = emb_s
        self.emb_t = emb_t

    def __call__(self, sample: dict) -> dict:
        tf_idx = sample["tf_idx"]
        target_idx = sample["target_idx"]

        sample["net_emb_tf_s"] = self.emb_s[tf_idx]
        sample["net_emb_tf_t"] = self.emb_t[tf_idx]
        sample["net_emb_target_s"] = self.emb_s[target_idx]
        sample["net_emb_target_t"] = self.emb_t[target_idx]
        return sample

