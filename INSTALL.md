# 安裝環境步驟

請依照以下步驟安裝所需環境：

1. 使用cmd 進入存放程式碼資料夾 `cd ...`
2. 建立虛擬環境：```python -m venv `venv` ```
3. 開啟虛擬環境
    * (Linux/MacOS)：```source `venv`/bin/activate```
    * (Windows)：````venv`\Scripts\activate```
    > ``venv``：可替換成你想要的虛擬環境名稱
4. **必須開啟cmd**，並進入環境：```.\`venv`\Scripts\activate```
5. 進入環境後，終端機上會顯示``(`venv`) ...``，請確認已進入環境
6. 更新並最佳化pip以及python必要套件版本：```pip install --upgrade pip setuptools wheel```
7. 安裝模型需求套件：```pip install -r requirements.txt```

> 若有需要額外安裝的套件，請使用 `pip install 套件名稱`後安裝

* 更新安裝套件清單 `pip freeze > requirements.txt`
