# -*- coding: utf-8 -*-
from fuzzysearch import find_near_matches
from fuzzywuzzy import process
import torch
import requests
import os
import copy
import boto3
import boto
import boto.s3
import sys
from boto.s3.key import Key

AWS_ACCESS_KEY_ID = 'J44Z76TED4DQ8TYQIDMF'
AWS_SECRET_ACCESS_KEY = '9BocByoSLMcBPvnf5e755e9PeA5oEgyo5t6MA9b4'

from flask import Flask, request, jsonify

app = Flask(__name__)

import json 
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import moviepy.editor as mp
from scipy.io.wavfile import write,read
import cv2
from pydub import AudioSegment
from pprint import pprint
import torch
import numpy as np
from facenet_pytorch import MTCNN
import shutil

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained

        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        return self.fc8(x)

total_clean = ('бакли рахман твейт корниш бреслин спенсер броди коупленд пилот карри палики тернер девгн рахман делон альда камминг рикман тудик рорвахер борштейн кингстон петтифер требек давалос сиддиг даддарио джонсон джонсон вудард фуглидоламс ничфил бахледайер гарфилд раннеллс скотт рихтер сэмберг барнард кинси хэтэуэй элгорт андерсон йелчин ассанте хаммер баттерфилд дженсен макдональд пэтридж джогиа новак херши гриллс фельдман маккензи стиллер уишоу лайвуд мэйклс уишоу грэнт мэлтон меккензи стиллер уишоу грэнт мэйтл мэлтон уишоу грэнт мэлтон пэйтл стивен мейер блетин глисон дэверн блессед деннехи мойнахан риган марлинг берк-чарвет бокслейтнер рейнольдс паундер маколифф кэмерон аккола паттон хеннеси фишер андервуд хардвуд аффлек уилсон петерсон мурино скорсон кроуфорд боузман чарис генсбур пальминтери хендлер макбрайд эджиофор беннет дикстра колфер хемсворт лоуэлл мартин мессина хендрикс риччи минц-плассе рамирес мерфи уильямс кертис доннелл циммер блеупер столл лайнер лайнеллис лайндел фанилофантини уэйанс эйкройд фоглер стивенс делани маккеллар отейл крейг кэмпбелл панабейкер дайер макбрайд крисс баутиста аттенборо кэссиди фаустино джунтоли хэревуд хенри джейсон кехнер кросс леттерман лайонс мамет мазуз моррисси уорнез уорнерич джоанерчез кросби якоби телер малруни китон крюгер клаттенхофф моргадо глисон купер риклс гловер сазерленд миллс мерфи джонс скотт кроес кэмерон уокер миллиган кэннон макдермотт моран макзин кэннон кэннон макдермотт моран вествок грифон кеннон кэннон кэссиди хендрикс томлинсон уоллах фаннинг бёрстин хашми маккен бреден маккормак робертс эстрада эндрюс боргнин хадсон моралес харрис дербез лонгория миллер ардант абрахам джонс ферги хендерсон уитакер раиса уильямсон хаймор агьеман пинто хедлунд маршалл бьюзи уллиел макфадден дэвис артертон аткинсон холливелл марини гуджайнс роганзей гуджайнс гуджайнс роганзей сулкин гервиг мбата-роу кристи штайнфельд спаркс симона спирритт циммер конструктор тредэуэй кейтель элизондо монтэг макатч бонэм кэвилл бертон суанк харпер грейнджер мандель бонневиль дэнси джекман пэрриш сомерхалдерз черный коулман пэрриш сомерхалдерз черный коулман пэрриш сомерхалдерз черный коулман пэрриш сомерхалдерз черный коулман давенпорт гриффин ультра-уэйтс ультрапорт ультрапорт ультрасовременный ультра-уэймон ультра-уайт-лайонс ультра-уэймон ультра-уэймон ультрасовременный деббуз значок космо кромвель фрейн лафферти липтон марсден маслоу несбитт пьюрфой родей сиглер крамер бадлер гарофало парриш мактир монтгомери дикинсон гаванкар тернер джонс харрис николь бэйтман кларк мэлсан смарт монтлес монтгомери дикинсон мэлсан мордлю смарт тамбор клемент лилливли льюис колеман эснер макинтош клэрнэдс клэрнэдс клэрнэдс слэрнэ систо феррара трейнор айзенберг маккартни меткалф спенсер уильямс лаундес скотт харрис майклс бродбент кэвизел джеффрис стерджесс фэллон аллен кьюсак риверс крупа ламли фроггатт фланиган эдгертон киннаман макхейл абрахам бэрроуман кэрролл гуджинск макхейл абрахам чоугетт хавролл гуджинск ларбетман ллойд мулани неттлз ноубл оливер рис-дэвис слэттери терри туртурро галецки фавро уэртас ловитц пертви джексон гаварис брюстер спаркс гарсия гордон-левитт бролин дюамель гробан хендерсон рэднор ндлин луи-дрейфус робертс стайлз сэндс николсон гонсало уолтерс эзарик ранаут брэйн бродер малфайн сообщество уолтерс эзарик ранаут фрэйн бродер малгеле сообщество уолтерсон эзарик ранаут брэнсис малфайн сообщество моралес харрис дербез лонгория миллер ардант абрахам джонс ферги хендерсон уитакер раиса уильямсон хаймор агьеман пинто хедлунд маршалл бьюзи уллиел макфадден дэвис артертон маршалл бьюзи уллиел макфадден дэвис артертон актёр бьюзи уллиел макфадден дэвис артертон ритнсон холливелл марини гуджайнсата ржалбай гуджайнсата рожир гуджайнсата ржалсей гуджайнсата рыжир гуджайнсата штайнфельд спаркс симона спирритт циммер конструктор тредэуэй кейтель элизондо монтэг макатч бонэм кэвилл бертон суанк харпер макпер грейнджер мандель бонневиль дэнси джекман лоринджер мандель бонневиль дэнси джекман лориффстон пэрриш грэнманн давид лориффон пэрриш сомерхорн мортон ультрамодерзайт ультрасовременный ультрасовременный ультразвук деббуз значок космо кромвель фрейн лафферти липтон марсден маслоу несбитт пьюрфой родей сиглер крамер бадлер гарофало парриш mcteer montgomery dickinson gavankar turner jones harris nicole batehring clarke smart тамбор клемент лилливли льюис колеман эснер макинтош клэрнэдс клэрнэдс клэрнэдс слэрнэ систо феррара трейнор айзенберг маккартни меткалф спенсер уильямс лаундес скотт харрис майклс бродбент кэвизел джеффрис стерджесс фэллон аллен кьюсак риверс крупа ламли фроггатт фланиган эдгертон киннаман макхейл абрахам бэрроуман кэрролл гуджинск макхейл абрахам чоугетт хавролл гуджинск ларбетман ллойд мулани неттлз ноубл оливер рис-дэвис слэттери терри туртурро галецки фавро уэртас ловитц пертви джексон гаварис брюстер спаркс гарсия гордон-левитт бролин дюамель гробан хендерсон рэднор ндлин луи-дрейфус робертс стайлз сэндс николсон гонсало макрос стайлз сэндс николсон гонсало уолтерс эзарик ранаут брэйн морган уолтерс эзарик ранаут фрэнс малофос уолтерс эзарик ранаут моргол сеффилс уолтерс эзарик ранаут фрэйн байнер мэлфен дженкинс макнамара винник робертсон джустен маккормик прескотт гриффин асельтон клири леклерк стивенс перри скоделарио новак аллен коннор митчелл берглунд гиддиш уильямс макдональд монако рейли рутерфорд уилтиш уиллиамс макдональд монако рейли рутерфорд уилтиш уилрион рогерс томпсон рогерс томпсон клэри клэри -biermann elise williams-paisley vangsness smit-mcphee jenner johnston schaal stewart alfonso sanon becker maclachlan richards minogue whitley turner benanti carmichael haddock crawford алонсо мишель салонга томпсон бродил хорнэнс мэнлинг мэнлинг эдельштейн кудроу ринна велчел каплан чабанол джонс девайн вербеек даймонд ферриньо бессон круикшанк грабил арназ лудакрис арнольд эванс ньюберри джилленхол акерман шерават гаммер патинкин дайал блукас форстер кросс мартиндейл робби бэмфорд каналс-баррера грация менунос шарапова авгеропулос хензепулос хеминг гамбс хемингард хэмингард уэблс кассулс хемингард хемингард хэмингард хэмингард клунс хендерсон маккатчеон паркер элизабет макдоннелл амальрик кассовиц бомер даллас лантер лиллард макфадьен модин моррисон адлер айронс шнайдер тиериот рудольф трейнер джордж маккарти питерман раучон сьюприк сьюприфт кьюдэри келли мэдсен мандо мухни шеннон тревино уэтерли конлин докери киган крусик монаган стаффорд уильямс кунис теллер сайрус йовович калинг фурлан косгроув ламберт ричардсон марголис бартон коллинз муссо рингуолд шеннон раймунд бладаккарин атиасл buring бьорлин посетитель картрайт харрис дормер мартинес портман макэлхон паркер эммануэль филлион бузолик нотон бониади макдонаф гейман грэйсс нелли фуртадо кэмпбелл хоран николс кролл минаж пельтц бехамер козер рэпен-уоллес эммануэль эммануэль шерзингер platt colman cooke thirlby уайлд уильямс хардвик чаплин исаак шрайбер консидайн лакшми brewster adlon соррентино чопра монахан dempsey duffy fugit суэйзи уилсон освальт джаматти gross mcgann reubens scheer сорвино уэсли jillette capaldi coyote фачинелли джексон jacobson гленистер диллер favino niney кумар монтгомери zinta принц тарантино dratch макадамс serbedzija родригес уилсон литтл мачио малек капур мэнтут уинстон макинтайр мейдер уилсон скотт обержонойс ретта дарби ифанс айоад хаммонд мэдден джерве родригес мэйалл ахмед брайдон корддри дирдек пичелл риггл амберт эндрю шерми ливингстон перлман роузи кеннеди макивер феган хантингтон-уайтли аткинсон маккланахан питерс хауэр джонс уилсон квантен макпартлин филлипп поттер эпата хокинс хьюэн палладио райми райли траммелл уотерстон уортингтон барктис уортингтон дэвис бернхард кабрера гилберт рамирес болджер гадон хайленд пэйлин полсон питерс бакула портер спидман коннери кэнан махер патрик пертви уильям рамамурти бергер макфарлейн мейерс роген капур вудли догерти копли винсессон мухаммед мурси винсэллсон агдашло раймс капур малхотра бэбетт гиллори бейкер каллоу эстин робинсон коппола капур оконедо болдман катич туччи пауэрс болдуин колберт манган рыцарь торговца бертон карелл харви джобс мартин бауэр шарлесбер вонтер моффат лёдер содер лёдер беркханг лёдер бертт лёдер моури пеникетт джонс тейлор хассан моури-хаусли грейг мэннинг маслани гевинсон лотнер момсен свифт дэнсон стэмп хэтчер ховард ньютон джеймс росси деккер гибсон хейден джейн леннон суинтон бертон дейли минчин сполл джонс лестон хупер кенни мисон селлек карран морган уайлдс сиван шакур пеннингтон шеридан блэкберн хэклин джеймс рэймонд адуба хадженс марсил феррес макклюр джастис роуэлл балан мортенсен гиллиган кассель мартелла пасторе пласидо мэдсен маттау гоггинс форвард-мартеллерс мартелла макклэрс болдуин фихтнер вендерс карсон харрельсон сэмюэл николь эфрон брафф галифианакис гордон квинто снайдер салдана'.split())


def fuzzy_extract(qs, ls, threshold):
    '''fuzzy matches 'qs' in 'ls' and returns list of 
    tuples of (word,index)
    '''
    for word, _ in process.extractBests(qs, (ls,), score_cutoff=threshold):
        #print('word {}'.format(word))
        for match in find_near_matches(qs, word, max_l_dist=1):
            match = word[match.start:match.end]
            #print('match {}'.format(match))
            index = ls.find(match)
            yield (match, index)

def censor_name(total_clean,transcript):
  censors = []
  for name in total_clean:
    finds = list(fuzzy_extract(name, ''.join(transcript.lower().split()),50))
    start_censor = [find[1] for find in finds]
    end_censor = [find[1] + len(name) for find in finds]
    censors.append((start_censor,end_censor))
  return censors

def process_audio(input_video,prefix):
  LANG_ID = "ru"
  MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
  SAMPLES = 5

  processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
  asr_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
  
  
  my_clip = mp.VideoFileClip(input_video)
  my_clip.audio.write_audiofile(r"my_result.mp3")
  timestamps = {}
  timestamps['result'] = {}
  times = []

  sound = AudioSegment.from_mp3("my_result.mp3")
  sound.export("outputfile.wav", format="wav", parameters=["-ar", "16000"])
  rate = 16000
  audio = np.array(sound.get_array_of_samples(),dtype = np.double)
  audio_edit = copy.copy(audio)


  model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=False)

  (get_speech_ts,
  get_speech_ts_adaptive,
  _, read_audio,
  _, _, _) = utils

  wav = read_audio('outputfile.wav')
  # adaptive way
  speech_timestamps = get_speech_ts_adaptive(wav, model)

  for speech_timestamp in speech_timestamps:
    start = speech_timestamp['start']
    end = speech_timestamp['end']
    audio_subset = audio[start:end]
    inputs = processor(audio_subset, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = asr_model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    predicted_sentences = predicted_sentences[0].lower()
    total_sentence_length = len(predicted_sentences)
    censors = censor_name(total_clean,predicted_sentences)
    for censor in censors:
            list_starts,list_ends = censor
            if list_starts is not None and list_ends is not None:
              for fragment in zip(list_starts,list_ends):


                audio_edit[start + int((end-start)*(fragment[0]/total_sentence_length)):start + int((end-start)*(fragment[1]/total_sentence_length))] = 0.0
                times.append({'time_start:':str((start + int((end-start)*(fragment[0]/total_sentence_length)))/16000),'time_end':str((start + int((end-start)*(fragment[1]/total_sentence_length)))/16000)})



  
  timestamps['result'] = (times)
  with open(prefix + '_audio'  + ".json", "w") as outfile:
    json.dump(timestamps, outfile)
  write("output.wav", 16000, audio_edit.astype(np.int16))


class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _blur(self,image,boxes_to_blur):
      for box in boxes_to_blur:
        print(box)
        image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = cv2.blur(image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] ,(50,50))
      return image        

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 255, 0),
                              thickness=2)

        except:
            pass

        return frame

    def run(self,input_file,prefix):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        model = VGG_16().double()

        timestamps = {}
        timestamps['result'] = {}
        times = []

        softmax_threshold = 0.5
        model.load_weights('VGG_FACE.t7')
        model.eval()
        cap = cv2.VideoCapture(input_file)
        frame_width  = int(cap.get(3))   
        frame_height = int(cap.get(4))
        v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(v_len)
        out = cv2.VideoWriter('video_output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame_width,frame_height))
        i=0
        for t in range(v_len):
            ret, frame = cap.read()
                # detect face box, probability and landmarks
            boxes, probs = self.mtcnn.detect(frame, landmarks=False)
            blur_boxes = []
                # drw on frame
            for box in boxes:
              boxed_face = cv2.resize(frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])],(224,224))
              im = torch.Tensor(boxed_face).permute(2, 0, 1).view(1, 3, 224, 224).double()
              im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).double().view(1, 3, 1, 1)
              preds = F.softmax(model(im), dim=1)
              print(np.max(preds.detach().numpy()[0,:],axis = -1))
              if np.max(preds.detach().numpy())>softmax_threshold:
                blur_boxes.append(box)


              
            for box in blur_boxes:
              times.append({'time_start:':str(t/25),'time_end':str((t+1)/25),'corner_1':[str(1920-box[0]),str(1080-box[3])],'corner_2':[str(1920-box[1]),str(1080-box[4])]})

            frame = self._blur(frame,blur_boxes)
            #frame = self._draw(frame, boxes, probs, landmarks)
            out.write(frame)
            i+=1
        timestamps['result'] = (times)
        with open(prefix + "_video.json", "w") as outfile:
          json.dump(timestamps, outfile)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/echo', methods=['POST'])
def hello():
   return jsonify(request.json)

@app.route('/recognize', methods=['GET', 'POST'])
def add_message():
    content = request.json
    url_to_download = content['source']
    print(url_to_download)
    prefix = content['prefix']
    print(prefix)

    response = requests.get(url_to_download, stream=True)
    with open('input.mp4', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    process_audio('input.mp4',prefix)
    mtcnn = MTCNN()
    fcd = FaceDetector(mtcnn)
    fcd.run('input.mp4',prefix)
    os.system("ffmpeg -i video_output.mp4 -i output.wav -c:v copy -map 0:v:0 -map 1:a:0 "+ prefix + "_result.mp4")

    session = boto3.session.Session()
    s3_endpoint_url = "https://obs.ru-moscow-1.hc.sbercloud.ru"
    client = session.client ( service_name="s3", 
    endpoint_url=s3_endpoint_url, 
    aws_access_key_id=AWS_ACCESS_KEY_ID, 
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
    use_ssl=False, 
    verify=False)

    for file in [prefix + '_audio.json',prefix + '_video.json',prefix + '_result.mp4']:
      client.upload_file(file, AWS_ACCESS_KEY_ID.lower(), file)
      
        


if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug = True, port=80)