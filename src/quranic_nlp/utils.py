import fnmatch
import os
import re


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        good_files = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in good_files)
    return results


POS_FA_UNI = {'اسم': 'NOUN',
              'فعل': 'VERB',
              'حرف': 'INTJ',
              'صفت': 'ADJ',
              'تصدیق': 'ADP',
              'قید': 'ADV',
              'کمکی': 'AUX',
              'ربط هماهنگ کننده': 'CCONJ',
              'تعیین کننده': 'DET',
              'عدد': 'NUM',
              'ذره': 'PART',
              'ضمیر': 'PRON',
              'اسم خاص': 'PROPN',
              'نقطه گذاری': 'PUNCT',
              'ربط تبعی': 'SCONJ',
              'نماد': 'SYM',
              'دیگر': 'X'}


POS_UNI_FA = {'NOUN': 'اسم',
              'VERB': 'فعل',
              'INTJ': 'حرف',
              'ADJ': 'صفت',
              'ADP': 'تصدیق',
              'ADV': 'قید',
              'AUX': 'کمکی',
              'CCONJ': 'ربط هماهنگ کننده',
              'DET': 'تعیین کننده',
              'NUM': 'عدد',
              'PART': 'ذره',
              'PRON': 'ضمیر',
              'PROPN': 'اسم خاص',
              'PUNCT': 'نقطه گذاری',
              'SCONJ': 'ربط تبعی',
              'SYM': 'نماد',
              'X': 'دیگر'}

AYEH_SEMANTIC = './data/qSyntaxSemantic/'
AYEH_SYNTAX = './data/syntax_data/'
QURAN_ORDER = './data/quarn_order.csv'

TRANSLATION = {
    'fa' : ['ansarian (Hussain Ansarian)', 'ayati (AbdolMohammad Ayati)', 'bahrampour (Abolfazl Bahrampour)', 'fooladvand (Mohammad Mahdi Fooladvand)', 'gharaati (Mohsen Gharaati)', 'ghomshei (Mahdi Elahi Ghomshei)', 'khorramdel (Mostafa Khorramdel)', "makarem (Baha'oddin Khorramshahi)", 'makarem (Naser Makarem Shirazi)', 'moezzi (Mohammad Kazem Moezzi)', 'mojtabavi (Sayyed Jalaloddin Mojtabavi)', 'sadeqi (Mohammad Sadeqi Tehrani)', 'safavi (Sayyed Mohammad Reza Safavi)'],
    'en' : ['ahmedali (Ahmed Ali)', 'ahmedraza (Ahmed Raza Khan)', 'arberry (A. J. Arberry)', 'daryabadi (Abdul Majid Daryabadi)', 'hilali (Muhammad Taqi-ud-Din al-Hilali and Muhammad Muhsin Khan)', 'itani (Talal Itani)', 'maududi (Abul Ala Maududi)', 'mubarakpuri (Safi-ur-Rahman al-Mubarakpuri)', 'pickthall (Mohammed Marmaduke William Pickthall)', 'qarai (Ali Quli Qarai)', 'qaribullah (Hasan al-Fatih Qaribullah and Ahmad Darwish)', 'sahih (Saheeh International)', 'sarwar (Muhammad Sarwar)', 'shakir (Mohammad Habib Shakir)', 'transliteration (English Transliteration)', 'wahiduddin (Wahiduddin Khan)', 'yusufali (Abdullah Yusuf Ali)'], 
    'sq' : ['nahi (Hasan Efendi Nahi)', 'mehdiu (Feti Mehdiu)', 'ahmeti (Sherif Ahmeti)'], 
    'ber' : ['mensur (Ramdane At Mansour)'], 
    'ar' : ['jalalayn (Jalal ad-Din al-Mahalli and Jalal ad-Din as-Suyuti)', 'muyassar (King Fahad Quran Complex)'], 
    'am' : ['sadiq (Muhammed Sadiq and Muhammed Sani Habib)'], 
    'az' : ['mammadaliyev (Vasim Mammadaliyev and Ziya Bunyadov)', 'musayev (Alikhan Musayev)'], 
    'bn' : ['hoque (Zohurul Hoque)' ,'bengali (Muhiuddin Khan)'], 
    'bs' : ['mlivo (Mustafa Mlivo)', 'korkut (Besim Korkut)'], 
    'bg' : ['theophanov (Tzvetan Theophanov)'], 
    'zh' : ['jian (Ma Jian)', 'majian (Ma Jian)'], 
    'cs' : ['hrbek (Preklad I. Hrbek)', 'nykl (A. R. Nykl)'], 
    'dv' : ['divehi (Office of the President of Maldives)'], 
    'nl' : ['keyzer (Salomo Keyzer)', 'leemhuis (Fred Leemhuis)', 'siregar (Sofian S. Siregar)'], 
    'de' : ['aburida (Abu Rida Muhammad ibn Ahmad ibn Rassoul)', 'bubenheim (A. S. F. Bubenheim and N. Elyas)', 'khoury (Adel Theodor Khoury)', 'zaidan (Amir Zaidan)'], 
    'ha' : ['gumi (Abubakar Mahmoud Gumi)'], 
    'hi' : ['farooq (Muhammad Farooq Khan and Muhammad Ahmed)', 'hindi (Suhel Farooq Khan and Saifur Rahman Nadwi)'], 
    'id' : ['indonesian (Indonesian Ministry of Religious Affairs)', 'jalalayn (Jalal ad-Din al-Mahalli and Jalal ad-Din as-Suyuti)', 'muntakhab (Muhammad Quraish Shihab et al.)'], 
    'it' : ['piccardo (Hamza Roberto Piccardo)'], 
    'jp' : ['japanese (Unknown)'], 
    'ko' : ['korean (Unknown)'], 
    'ms' : ['basmeih (Abdullah Muhammad Basmeih)'], 
    'ku' : ['asan (Burhan Muhammad-Amin)'], 
    'ml' : ['abdulhameed (Cheriyamundam Abdul Hameed and Kunhi Mohammed Parappoor)', 'karakunnu (Muhammad Karakunnu and Vanidas Elayavoor)'], 
    'no' : ['breg (Einar Berg)'], 
    'ps' : ['abdulwali (Abdulwali Khan)'], 
    'ro' : ['grigore (grigore)'], 
    'pt' : ['elhayek (Samir El-Hayek)'], 
    'pl' : ['bielawskiego (Józefa Bielawskiego)'], 
    'ru' : ['abuadel (Abu Adel)', 'muntahab (Ministry of Awqaf, Egypt)', 'krachkovsky (Ignaty Yulianovich Krachkovsky)', 'kuliev (Elmir Kuliev)', "kuliev-alsaadi (Elmir Kuliev (with Abd ar-Rahman as-Saadi's commentaries)", 'osmanov (Magomed-Nuri Osmanovich Osmanov)', 'porokhova (V. Porokhova)', 'sablukov (Gordy Semyonovich Sablukov)'], 
    'sd' : ['amroti (Taj Mehmood Amroti)'], 
    'so' : ['abduh (Mahmud Muhammad Abduh)'], 
    'es' : ['bornez (Raúl González Bórnez)', 'cortes (Julio Cortes)', 'garcia (Muhammad Isa García)'], 
    'sv' : ['bernstrom (Knut Bernström)'], 
    'sw' : ['barwani (Ali Muhsin Al-Barwani)'], 
    'tg' : ['ayati (AbdolMohammad Ayati)'], 
    'ta' : ['tamil (Jan Turst Foundation)'], 
    'tt' : ['nugman (Yakub Ibn Nugman)'], 
    'th' : ['thai (King Fahad Quran Complex)'], 
    'tr' : ['golpinarli (Abdulbaki Golpinarli)', 'bulac (Alİ Bulaç)', 'transliteration (Muhammet Abay)', 'diyanet (Diyanet Isleri)', 'vakfi (Diyanet Vakfi)', 'yuksel (Edip Yüksel)', 'yazir (Elmalili Hamdi Yazir)', 'ozturk (Yasar Nuri Ozturk)', 'yildirim (Suat Yildirim)', 'ates (Suleyman Ates)'], 
    'uz' : ['sodik (Muhammad Sodik Muhammad Yusuf)'], 
    'ug' : ['saleh (Muhammad Saleh)'], 
    'ur' : ["maududi (Abul A'ala Maududi)", 'kanzuliman (Ahmed Raza Khan)', 'ahmedali (	Ahmed Ali)', 'jalandhry (Fateh Muhammad Jalandhry)', 'qadri (Tahir ul Qadri)', 'jawadi (Syed Zeeshan Haider Jawadi)', 'junagarhi (Muhammad Junagarhi)', 'najafi (Muhammad Hussain Najafi)'], 
    }

def get_sim_ayahs(soure, ayeh):
    output = []
    with open('./data/sim_ayat.txt') as file:
        sims = file.readlines()
        for sim in sims:
            ayeha = sim.split('\t')
            so = int(ayeha[0][:-3])
            ay = int(ayeha[0][-3:])
            if soure == so and ay == ayeh:
                for aye in ayeha[1:]:
                    temp , cost = aye.split(':')
                    output.append(str(int(temp[:-3])) + '#' + str(int(temp[-3:])))
    return output

def get_translations(input, soure, ayeh):
    if '#' in input:
        lang, index = input.split('#')
        name = TRANSLATION[lang][int(index)].split()[0]
        filename = os.path.join('./data/translate_quran', lang, name+'.txt')
        with open(filename) as file:
            txt = file.read()
        tempAyeh = ayeh
        if soure == 1:
            tempAyeh += 1
        start = re.search(f"{soure}\|{tempAyeh}\|", txt)
        end = re.search(f"{soure}\|{tempAyeh+1}\|", txt)
        
        if end != None:
            return txt[start.end():end.start()]
        else:
            end2 = re.search(f"{soure+1}\|{1}\|", txt)
            if end2 != None:
                return txt[start.end():end2.start()]
            else:
                return txt[start.end():].split('\n')[0]
    else:
        txt_traslation = []
        for names in TRANSLATION[lang]:
            name = names.split()[0]
            filename = os.path.join('./data/translate_quran', lang, name+'.txt')
            with open(filename) as file:
                txt = file.read()
            start = re.search(f"{soure}\|{ayeh+1}\|", txt)
            end = re.search(f"{soure}\|{ayeh+2}\|", txt)
            if end != None:
                txt_traslation.append(txt[start.end():end.start()])
            else:
                txt_traslation.append(txt[start.end:])
        return txt_traslation


def print_all_translations():
    for key, value in TRANSLATION.items() :
        for val in value:
            # val = val.split('(')[0].strip()
            val = val.split('(')[1].split(')')[0].strip()
            print (key, val)

# AYEH_INDEX = ['حمد',
#  'بقره',
#  'ال عمران',
#  'نساء',
#  'مااده',
#  'انعام',
#  'اعراف',
#  'انفال',
#  'توبه',
#  'يونس',
#  'هود',
#  'يوسف',
#  'رعد',
#  'ابراهيم',
#  'حجر',
#  'نحل',
#  'اسراء',
#  'كهف',
#  'مريم',
#  'طه',
#  'انبياء',
#  'حج',
#  'مومنون',
#  'نور',
#  'فرقان',
#  'شعراء',
#  'نمل',
#  'قصص',
#  'عنكبوت',
#  'روم',
#  'لقمان',
#  'سجده',
#  'احزاب',
#  'سبا',
#  'فاطر',
#  'يس',
#  'صافات',
#  'ص',
#  'زمر',
#  'غافر',
#  'فصلت',
#  'شوري',
#  'زخرف',
#  'دخان',
#  'جاثيه',
#  'احقاف',
#  'محمد',
#  'فتح',
#  'حجرات',
#  'ق',
#  'ذاريات',
#  'طور',
#  'نجم',
#  'قمر',
#  'رحمان',
#  'واقعه',
#  'حديد',
#  'مجادله',
#  'حشر',
#  'ممتحنه',
#  'صف',
#  'جمعه',
#  'منافقون',
#  'تغابن',
#  'طلاق',
#  'تحريم',
#  'ملك',
#  'قلم',
#  'حاقه',
#  'معارج',
#  'نوح',
#  'جن',
#  'مزمل',
#  'مدثر',
#  'قيامت',
#  'انسان',
#  'مرسلات',
#  'نبا',
#  'نازعات',
#  'عبس',
#  'تكوير',
#  'انفطار',
#  'مطففين',
#  'انشقاق',
#  'بروج',
#  'طارق',
#  'اعلي',
#  'غاشيه',
#  'فجر',
#  'بلد',
#  'شمس',
#  'ليل',
#  'ضحي',
#  'شرح',
#  'تين',
#  'علق',
#  'قدر',
#  'بينه',
#  'زلزله',
#  'عاديات',
#  'قارعه',
#  'تكاثر',
#  'عصر',
#  'همزه',
#  'فيل',
#  'قريش',
#  'ماعون',
#  'كوثر',
#  'كافرون',
#  'نصر',
#  'مسد',
#  'اخلاص',
#  'فلق',
#  'ناس']

AYEH_INDEX = [['حمد', 'الفاتحه', 'ام القران', 'فاتحه الكتاب'],
 ['بقره', 'سنام القران', 'فسطاط القران'],
 ['ال عمران', 'طيبه', 'دبابيج'],
 ['نساء', ],
 ['مااده', 'منقذه', 'عقور'],
 ['انعام'],
 ['اعراف', 'مص'],
 ['انفال', 'بدر'],
 ['توبه', 'فاضحه', 'منقره', 'بحوث', 'برااه', 'حاضره', 'مثيره'],
 ['يونس'],
 ['هود'],
 ['يوسف'],
 ['رعد'],
 ['ابراهيم'],
 ['حجر'],
 ['نحل', 'نعم'],
 ['اسراء', 'سبحان', 'بني اسراايل'],
 ['كهف', 'حااله'],
 ['مريم'],
 ['طه', 'كليم', 'حكيم'],
 ['انبياء'],
 ['حج'],
 ['مومنون'],
 ['نور'],
 ['فرقان'],
 ['شعراء', 'جامعه'],
 ['نمل', 'سليمان'],
 ['قصص'],
 ['عنكبوت'],
 ['روم'],
 ['لقمان'],
 ['سجده', 'تنزيل', 'م', 'جزر', 'مضاجع', 'سجده لقمان' ],
 ['احزاب'], 
 ['سبا'], 
 ['فاطر', 'ملااكه'], 
 ['يس', 'ريحانه القران', 'قلب القران'],  
 ['صافات'],
 ['ص'],  
 ['زمر', 'غرف'],  
 ['غافر', 'طول', 'مومن'],  
 ['فصلت', 'مصابيح', 'حم السجده'],  
 ['شوري', 'حمعسق'],  
 ['زخرف'],  
 ['دخان'],  
 ['جاثيه', 'شريعه'],  
 ['احقاف'],  
 ['محمد', 'قتال'],  
 ['فتح'],  
 ['حجرات'],  
 ['ق', 'باسقات'],  
 ['ذاريات'],  
 ['طور'],  
 ['نجم'],  
 ['قمر', 'اقتربت الساعه'],  
 ['رحمان'],  
 ['واقعه'],  
 ['حديد'],  
 ['مجادله', 'ظهار'],  
 ['حشر', 'بني النضير'],  
 ['ممتحنه', 'موده', 'امتحان'],  
 ['صف', 'عيسي', 'حواريين'],  
 ['جمعه'],  
 ['منافقون'],  
 ['تغابن'],  
 ['طلاق', 'نساء القصري', 'نساء الصغري'],  
 ['تحريم', 'يا ايها النبي', 'متحرم', 'لم تحرم'],  
 ['ملك', 'مانعه', 'تبارك', 'منحيه', 'واقعيه'],  
 ['قلم', 'ن'],  
 ['حاقه'],  
 ['معارج', 'واقع', 'سال ساال'],  
 ['نوح'],  
 ['جن'],  
 ['مزمل'],  
 ['مدثر'],  
 ['قيامت'],  
 ['انسان', 'دهر', 'ابرار', 'هل اتي'],  
 ['مرسلات', 'عرف'],  
 ['نبا', 'معصرات', 'عم', 'تساول'],  
 ['نازعات'],  
 ['عبس', 'اعمي', 'سفره'],  
 ['تكوير', 'كورت'],  
 ['انفطار', 'انفطرت'],  
 ['مطففين', 'تطفيف'],  
 ['انشقاق'],  
 ['بروج', 'پيامبران'],  
 ['طارق'],  
 ['اعلي'],  
 ['غاشيه'],  
 ['فجر', 'حسين'],  
 ['بلد'],  
 ['شمس', 'ناقه', 'صالح'],  
 ['ليل'],
 ['ضحي'],
 ['شرح', 'م نشرح', 'انشراح'],  
 ['تين'],  
 ['علق'],  
 ['قدر', 'انا انزلناه'],  
 ['بينه', 'اهل الكتاب', 'قيامه', 'بريه', 'لم يكن'],  
 ['زلزله', 'زلزال'],  
 ['عاديات'],  
 ['قارعه'],  
 ['تكاثر'],  
 ['عصر'],  
 ['همزه', 'لمزه'],  
 ['فيل', 'م تر كيف'],  
 ['قريش', 'ايلاف'],  
 ['ماعون', 'دين', 'ارايت'],  
 ['كوثر'],  
 ['كافرون', 'مقشقشه', 'عباده', 'جحد'],  
 ['نصر', 'توديع'],  
 ['مسد', 'تبت', 'لهب', 'ابي لهب'],  
 ['اخلاص', 'توحيد', 'اساس', 'صمد', 'نسبه رب'],  
 ['فلق'],  
 ['ناس']
 ]