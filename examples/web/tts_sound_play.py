import os
import sys
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import torch
import sounddevice as sd
from tools.audio import float_to_int16
import ChatTTS
import threading
from queue import Queue
import time

class TTSSoundPlay:
    def __init__(self, play_queue):
        self.play_queue = play_queue  # 播报任务队列

    def synthesize_and_queue(self, texts):
        chat = ChatTTS.Chat()
        chat.load(compile=True)  # 加载TTS模型
        texts = [text + "[uv_break]" for text in texts]  # 在每个元素后添加 "[uv_break]"
        print(texts)

        # 采样一个说话人
        rand_spk = "蘁淰敱欀搄圲寏烻瑾膘啑晩縤拞蒁俇赯異娶峕妰硪值聍胻刋俫妕行識笹嫘瘑苳歪謔荬晞忸呬楁沿綊噫趧臨賏牷璥格彥瀕挹怏灤橙脧薭譛蓋籭棼胰倐媑槧爡藂賷甯剺瑟恢牋砙弚晱機滛倩瘜療浚捥岠嶎呋敂墊玌沅栿怹幕磥咾珦嘆膶澫怖搡暇幪秾緈賹柃偏堤忆爬漌覆給朘箰幊囧爼哰嫬菺碛搶厔缯敐艎诱刏葷咂哞笃囈揅起线欱岐趪勲憎羓肓虍犩枬紬炝瞵腇瀢童劚恕譌峬祦煨捑腍蜸肿脃慫谙膑橂咒艸儮埵羑叨凜嬟奄澤珉荽徾碷依泒蟨夃悑曔啮悻咠跄亶己梦撛埳桟疇舂峑褚悴欁廉羀糃箽蟸婶燑嗺裶壚烵翓肟嗑襲豑縧恵婻蘒谡瑟墽焈蚲舥丽脼篙柪薗苣燫垰讀懢嗲窙尺噟圪磶襖薂竘哼稊叟睗諣揫椘祕县穜奺侔弁渖哰爀犇筀撮孲荵咠蘿蝣猗谹蕻绉乡甲矾蒡足浲檊墢幆讽谻燁呗憉榬櫆拦岨兌溕縗怩叚労淖懩戣烕態慮瀚籏薗委艺禆蠁襡堮悡揞扅蘉窗傽亁曥枡矜筩朆瀈荓櫅匌勩帪沄穝嘟虵硷趥尶琮侩濇论婩缯胙襠外笮搻僋賚絕娊语熉聡蓧糼塲牼玎碾秗尺繣独癥貐惰夈埋撑螨屨蚆纘伳屆怭廖氹袗蟼珫姁觊篣续夑嫨壾奉槔俏炥嚔虶峠褐蚱妣莥剮貅湑搶惩廝矲趟嚫琼峩畬神彊呈款獫硕帩肥磀袘坬绂諍讙觫蒿欪磌葎杫昪戉艓炮劋甤屌摖爜粡槣菉憉瘫磖嬒烿凸帎壃启偎绺褈抨貉殜聀濯睓卼繚罸澝咆噳寋崯囅敌殬廈慁腾姅蟏斡覨耮睪罽砚罯喏壠熘敛兰塛庨濶糣疈惮嚕琸蟊兀蟙班玁爂捆徶拥桁纩嗝渰岄耮犾筕褅張毭煹緘戮掻芻蔫玶咅瘢奧苼缟懮塂毟孶槽笺嶤眥殺暃彊蒚憏枿緷佰游俦僀繒汿貪桕詘胥蔝曚猼襳槙敖訇嗺擭卼甪藩呾皃晳攨叩嶼析甙傛虪什尬氯圛媍网訇乣枆彟梕燮磧荙琂屐洼莡觕埶喝矲旼跗涠憓焪蚾坟芘炫奓蝌罈豂匊贩啩沼簟瀍犷万菨譚趁暙浿癚玱嗟藯窌滲晑貀棜嘿冨讝燭貰亀涼敒獧蘪窜漣棨攻悵拦芠宥昜罢褮煓烞孯捤赶腥掝谆炔橪嫮寊澶櫘肮澔訰拉篵慿氡岵俍滟橭税抖滈蚘趤椎纙淉砽喋賌缮慂常竞忬磳泰聠绶撃渙嶊聄搁腓唄焰傏稘渳佀氭悲蜛涷脣乯袨箼狿琵臭儥刊蓱綖乖柸诓缯货癩縝蕬貟氅誺拗洑彫仓壠凴矂痞毽趀剒皌趼虾檪殿巌十燍橄夔覿蠔庨捒罉甈椀歇纯緋忺夸緁呙奎畦将石蕙嬘嫧让樁护厰玩囵襧吤垁翁蓆俩咮昽攍垭廌温襐窂貧制榟櫲秢媍夓催嘻党皟探准椽嵝柄燍囖掷臥腞嗝叾砚咖噣擴囜溤噄姆愋蒊嫀繡尉亵擈蒞撕昘媃瑪嫪悆噢忆皛媨蛎耴訨壣眸渀㴃"

        # 设置推理参数
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            temperature=0.3,
            top_P=0.7,
            top_K=20,
        )

        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_4]',
        )

        # 执行文本到语音合成
        wavs = chat.infer(
            texts,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )

        # 将合成的音频数据转换为 int16 格式
        audio_data = float_to_int16(wavs[0]).T

        # 将音频任务 (audio_data, 24000) 放入队列，等待播放
        self.play_queue.put((audio_data, 24000))

def audio_player(play_queue, stop_event):
    while not stop_event.is_set() or not play_queue.empty():
        if not play_queue.empty():
            audio_data, samplerate = play_queue.get()
            # 播放音频
            sd.play(audio_data, samplerate)
            sd.wait()  # 等待播放完成
        else:
            time.sleep(1)  # 如果队列为空，等待一段时间再检查
    print("Audio playback finished.")

if __name__ == "__main__":
    play_queue = Queue()
    stop_event = threading.Event()  # 创建停止事件

    # 创建 TTS 播报对象
    tts = TTSSoundPlay(play_queue)

    # 创建并启动音频播放线程
    player_thread = threading.Thread(target=audio_player, args=(play_queue, stop_event))
    player_thread.start()

    # 输入示例文本
    example_texts = ["早上好，主人，今天要吃点什么呢"]
    # 调用函数合成并将音频任务推入队列
    tts.synthesize_and_queue(example_texts)

    # 设置停止事件，标志音频播放线程可以结束
    stop_event.set()
    player_thread.join()  # 再次等待音频播放线程结束
    print("All tasks completed, exiting.")
