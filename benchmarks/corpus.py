"""Synthetic bilingual acceptance corpus, not a claim of real-world quality.

Each scenario has two parallel documents and five independently phrased queries.
Three queries are also tested across language-filtered collections (8 per group).
Translations and related queries always share a split. Review labels before using
this corpus to select production models; it complements, not replaces, user data.
"""

# English passage, Arabic passage, two EN / two MSA / one Egyptian query.
SCENARIOS = [
    ("People kept leaving their badges by the kettle. The remaining staff inherited their work. Nobody wanted another pizza evening; they wanted their Sundays back.",
     "ترك الزملاء بطاقاتهم بجوار الغلاية ورحلوا. تراكمت مهامهم على الباقين. لم يرغب أحد في حفلة جديدة؛ أرادوا قضاء يوم الجمعة مع أسرهم.",
     ["employee burnout and retention", "workplace perks cannot fix excessive workloads", "استقالات بسبب الإرهاق الوظيفي", "ضغط العمل وفقدان التوازن مع الحياة", "الناس بتمشي من الشغل عشان الضغط"]),
    ("Every midnight the server copies changed records to a second building. Once a month we restore a copy on a spare machine and compare the totals.",
     "ينسخ الخادم السجلات المتغيرة كل منتصف ليلة إلى مبنى آخر. نستعيد نسخة على جهاز احتياطي شهريا ونتحقق من تطابق الأرقام.",
     ["disaster recovery verification", "recovering business data after a server failure", "اختبار استعادة البيانات بعد كارثة", "حماية السجلات من تعطل الخادم", "لو السيرفر وقع نرجع الداتا إزاي"]),
    ("The same customer paid invoice 804 twice. Accounting will return the second transfer to its original account after matching both receipts.",
     "سدد العميل الفاتورة ٨٠٤ مرتين. سيعيد قسم الحسابات التحويل الثاني إلى الحساب الأصلي بعد مطابقة الإيصالين.",
     ["refund for a duplicate payment", "invoice 804 charged twice", "استرداد مبلغ مكرر للفاتورة 804", "عميل دفع نفس الفاتورة مرتين", "دفعت الفاتورة مرتين وعايز فلوسي"]),
    ("Visitors cannot reach the reception desk without climbing three steps. The proposed entrance replaces them with a gently sloping surface and a wider doorway.",
     "لا يصل الزوار إلى الاستقبال دون صعود ثلاث درجات. يستبدل التصميم المقترح الدرج بممر مائل قليلا وباب أوسع.",
     ["wheelchair access improvements", "making the building accessible", "تهيئة المدخل لذوي الإعاقة", "وصول مستخدمي الكراسي المتحركة", "واحد على كرسي متحرك يدخل المبنى إزاي"]),
    ("The pot contains chickpeas, lemon, sesame paste and olive oil. No dairy is added. The sesame ingredient must still be disclosed to guests.",
     "يحتوي الطبق على الحمص والليمون والطحينة وزيت الزيتون. لا نضيف الحليب أو مشتقاته، لكن يجب إبلاغ الضيوف بوجود السمسم.",
     ["dairy free food with an allergen warning", "sesame in a vegan dish", "طعام خال من الألبان يحتوي على السمسم", "تحذير حساسية الطحينة", "الأكلة دي من غير لبن بس فيها سمسم"]),
    ("A caller claimed to be from payroll and asked Lina for the six digits sent to her phone. She hung up and contacted the internal help desk directly.",
     "ادعى متصل أنه من قسم الرواتب وطلب من لينا الأرقام الستة التي وصلت إلى هاتفها. أغلقت المكالمة واتصلت بالدعم الداخلي بنفسها.",
     ["social engineering using verification codes", "protecting an account from phone scams", "احتيال لسرقة رمز التحقق", "التعامل مع انتحال هوية قسم الرواتب", "حد كلمني وطلب كود الموبايل"]),
    ("Indoor readings rose whenever the windows stayed shut during meetings. Opening two opposite windows between sessions brought the readings down.",
     "ارتفعت القراءات داخل الغرفة عندما بقيت النوافذ مغلقة أثناء الاجتماعات. أدى فتح نافذتين متقابلتين بين الجلسات إلى انخفاضها.",
     ["improving meeting room ventilation", "stuffy office air", "تحسين تهوية قاعات الاجتماعات", "تجديد الهواء في المكاتب", "الأوضة مكتومة نفتح الشبابيك"]),
    ("The owner may enter the flat only after giving the occupant two days of notice, except when water is escaping or lives are in immediate danger.",
     "لا يحق للمالك دخول الشقة إلا بعد إخطار المقيم قبل يومين، باستثناء تسرب المياه أو وجود خطر مباشر على الأرواح.",
     ["tenant privacy and landlord entry", "advance notice before property inspection", "خصوصية المستأجر عند دخول المالك", "مدة الإخطار قبل تفتيش الشقة", "صاحب الشقة ينفع يدخل من غير ما يقولي"]),
    ("The orchard flowered earlier this spring, but a cold night damaged most blossoms. Farmers placed temperature sensors near the lowest rows for the next season.",
     "أزهرت الأشجار مبكرا هذا الربيع، لكن ليلة شديدة البرودة أتلفت معظم الأزهار. وضع المزارعون حساسات حرارة قرب الصفوف المنخفضة للموسم المقبل.",
     ["protecting crops from late frost", "cold weather damage during flowering", "أثر الصقيع على أزهار الفاكهة", "مراقبة البرودة لحماية المحاصيل", "البرد بوظ زهر الشجر"]),
    ("A courier scanned the parcel as delivered, but the photograph showed a blue door. Our customer's door is green. We asked the depot to check the adjacent street.",
     "سجل المندوب تسليم الطرد، لكن الصورة أظهرت بابا أزرق وباب العميل أخضر. طلبنا من المستودع التحقق من الشارع المجاور.",
     ["package delivered to the wrong address", "disputing a delivery confirmation", "تسليم شحنة إلى عنوان خاطئ", "الاعتراض على إثبات استلام طرد", "الطلب مكتوب وصل بس مش عندي"]),
    ("New starters receive a temporary account that expires after seven days. Their manager must request named access before that period ends; shared passwords are prohibited.",
     "يحصل الموظفون الجدد على حساب مؤقت ينتهي بعد سبعة أيام. يجب على المدير طلب صلاحيات شخصية قبل انتهاء المدة ويمنع تبادل كلمات المرور.",
     ["secure employee onboarding", "temporary access expiration policy", "إدارة صلاحيات الموظفين الجدد", "انتهاء الحساب المؤقت بعد أسبوع", "أنا لسه بادئ والحساب هيقفل"]),
    ("The workshop alternates spoken explanation with written captions. Participants can follow every demonstration without relying on the speaker's voice.",
     "تجمع الورشة بين الشرح الشفهي والنص المكتوب على الشاشة. يستطيع المشاركون متابعة كل عرض دون الاعتماد على صوت المتحدث.",
     ["training accessibility for deaf participants", "captions during live demonstrations", "إتاحة التدريب لضعاف السمع", "نصوص مصاحبة للشرح الصوتي", "عايز أفهم الشرح من غير ما أسمع"]),
    ("After sunset the panels produce nothing. Stored daytime output supplies the lights until morning; the controller reserves a fifth of the capacity for outages.",
     "لا تنتج الألواح شيئا بعد الغروب. تغذي الطاقة المخزنة نهارا المصابيح حتى الصباح وتحتفظ وحدة التحكم بخمس السعة لانقطاع الشبكة.",
     ["solar battery storage at night", "reserve power for electricity outages", "تخزين الطاقة الشمسية للاستخدام الليلي", "احتياطي الكهرباء عند انقطاع الشبكة", "نشغل النور بالليل من طاقة النهار"]),
    ("We keep applications for six months after a vacancy closes. Then identifying fields are removed, while anonymous counts remain for annual reporting.",
     "نحتفظ بطلبات التوظيف ستة أشهر بعد إغلاق الشاغر ثم نحذف الحقول التي تكشف الهوية ونبقي أعدادا مجهولة الهوية للتقرير السنوي.",
     ["applicant data retention and anonymization", "deleting personal information after recruitment", "مدة الاحتفاظ ببيانات المتقدمين", "إخفاء الهوية في سجلات التوظيف", "بيانات اللي قدموا شغل بتتمسح إمتى"]),
    ("Each afternoon a volunteer visits residents who live alone. If nobody answers, the volunteer phones their chosen contact rather than assuming they are away.",
     "يزور متطوع كل عصر السكان الذين يعيشون بمفردهم. إذا لم يرد أحد يتصل بالشخص الذي اختاره المقيم بدلا من افتراض أنه خارج المنزل.",
     ["welfare checks for isolated residents", "community support for people living alone", "الاطمئنان على المقيمين بمفردهم", "دعم اجتماعي لمن يعيش وحيدا", "مين يطمن على الناس اللي عايشة لوحدها"]),
    ("Before release, two colleagues independently compare the translated instructions with the original. They pay special attention to quantities and words that reverse meaning.",
     "يراجع زميلان التعليمات المترجمة كل على حدة بمقارنتها بالأصل. يركزان على الكميات والكلمات التي تقلب المعنى قبل النشر.",
     ["translation quality assurance", "checking negation and numbers in translated manuals", "ضمان جودة الترجمة", "مراجعة النفي والأرقام في التعليمات", "نتأكد إزاي إن الترجمة مغيرتش المعنى"]),
    ("Passengers who cannot complete the journey may request the unused portion of their fare. Tickets bought through an agent must be handled by that same agent.",
     "يستطيع المسافر الذي لا يكمل الرحلة طلب قيمة الجزء غير المستخدم من الأجرة. تعالج التذاكر المشتراة عبر وسيط بواسطة الوسيط نفسه.",
     ["partial travel ticket refunds", "unused journey reimbursement through an agent", "استرداد قيمة الجزء غير المستخدم من التذكرة", "إجراءات رد الأجرة عن طريق الوسيط", "مكملتش الرحلة أرجع باقي الفلوس إزاي"]),
    ("The application works when the network is unavailable. Edits are stored on the device, then uploaded when a connection returns. Conflicting edits are shown for review.",
     "يعمل التطبيق دون اتصال بالشبكة. تحفظ التعديلات على الجهاز ثم ترفع عند عودة الاتصال وتعرض التعديلات المتعارضة للمراجعة.",
     ["offline synchronization and conflict resolution", "editing without an internet connection", "مزامنة التعديلات بعد العمل دون اتصال", "حل تعارض نسخ البيانات", "أعدل من غير نت ولما يرجع يرفع التعديل"]),
    ("The new crossing gives pedestrians ten extra seconds. A raised surface slows approaching cars near the school gate at the beginning and end of the day.",
     "يمنح المعبر الجديد المشاة عشر ثوان إضافية. يخفف سطح مرتفع سرعة السيارات قرب بوابة المدرسة عند بدء اليوم وانتهائه.",
     ["road safety for school children", "traffic calming near a pedestrian crossing", "سلامة الأطفال عند عبور الطريق", "تهدئة المرور بجوار المدارس", "نخلي العيال تعدي الشارع بأمان"]),
    ("We discovered the same expense under two different project codes. Future submissions will be compared by receipt number, amount and date before approval.",
     "وجدنا المصروف نفسه تحت رمزين مختلفين للمشروعات. ستقارن الطلبات المقبلة برقم الإيصال والمبلغ والتاريخ قبل الموافقة.",
     ["preventing duplicate expense reimbursement", "detecting repeated receipts across projects", "منع صرف المصروفات مرتين", "اكتشاف الإيصالات المكررة", "نفس الإيصال اتحاسب مرتين"]),
    ("Students may borrow a laptop for the whole term. A broken charger can be exchanged at the library desk without replacing the computer or ending the loan.",
     "يمكن للطلاب استعارة حاسوب محمول طوال الفصل الدراسي. يستبدل الشاحن المعطل في مكتب المكتبة دون تغيير الجهاز أو إنهاء الإعارة.",
     ["student laptop lending scheme", "replacing a charger on borrowed equipment", "إعارة الحواسيب للطلاب", "استبدال شاحن جهاز مستعار", "شاحن اللابتوب اللي مستلفه باظ"]),
    ("The lift motor is inspected every quarter. An unusual vibration triggers an immediate shutdown until a technician checks the bearings, even if the doors still open normally.",
     "يفحص محرك المصعد كل ثلاثة أشهر. تؤدي الاهتزازات غير المعتادة إلى إيقافه فورا حتى يفحص الفني المحامل حتى لو ظلت الأبواب تعمل بصورة طبيعية.",
     ["preventive elevator maintenance", "shutting down machinery after abnormal vibration", "الصيانة الوقائية للمصاعد", "إيقاف الآلات عند ظهور اهتزاز غير معتاد", "الأسانسير بيتهز نوقفه ولا لأ"]),
    ("Attendees may choose a quiet room with dim lighting instead of the crowded hall. The same session is streamed there, and leaving early requires no explanation.",
     "يستطيع الحاضرون اختيار غرفة هادئة بإضاءة خافتة بدلا من القاعة المزدحمة. تبث الجلسة نفسها هناك ولا يحتاج المغادر مبكرا إلى تقديم تفسير.",
     ["sensory friendly event accommodations", "alternatives to noisy crowded conference rooms", "تهيئة الفعاليات للحساسية الحسية", "توفير مكان هادئ في المؤتمرات", "الدوشة والزحمة بتتعبني في المؤتمر"]),
    ("The committee publishes the criteria before receiving proposals. Members with a family connection to a bidder leave the room while that proposal is discussed.",
     "تنشر اللجنة المعايير قبل استلام العروض. يغادر العضو الذي تربطه صلة قرابة بأحد مقدمي العروض الغرفة أثناء مناقشة عرضه.",
     ["conflict of interest in procurement", "fair evaluation of competing bids", "تضارب المصالح في المشتريات", "حياد تقييم العروض التجارية", "واحد في اللجنة قريب صاحب العرض"]),
]

NO_ANSWER = [
    "orbital mechanics of a lunar lander", "أسعار العملات المشفرة غدا",
    "the captain's private bank password", "كلمة مرور حساب المدير",
    "dinosaur fossils in Antarctica", "كيفية صناعة الزجاج البركاني",
    "results of the 2035 world cup", "من فاز بكأس العالم عام ٢٠٣٥",
    "quantum error correcting surface codes", "تصحيح الأخطاء في الحوسبة الكمية",
    "recipe for strawberry cheesecake", "طريقة إعداد كعكة الفراولة بالجبن",
    "underwater volcano eruption timetable", "موعد ثوران بركان تحت البحر",
    "my grandmother's birth certificate", "شهادة ميلاد جدتي",
]


def materialize(root):
    root.mkdir(parents=True, exist_ok=True)
    queries = []
    for number, (english, arabic, questions) in enumerate(SCENARIOS):
        names = [f"record-{number:03d}.txt", f"record-{number:03d}.md"]
        for name, text in zip(names, (english, arabic)):
            (root / name).write_text(text, encoding="utf-8")
        split = "heldout" if number % 4 == 3 else "development"
        for position, query in enumerate(questions):
            language = "en" if position < 2 else "egy" if position == 4 else "ar"
            queries.append(dict(query=query, relevant=names, group=number, split=split, language=language))
        for position in (0, 2, 4):
            target = names[1] if position == 0 else names[0]
            direction = "en→ar" if position == 0 else "egy→en" if position == 4 else "ar→en"
            extension = "md" if position == 0 else "txt"
            queries.append(dict(query=questions[position] + " type:" + extension,
                                relevant=[target], group=number, split=split, language=direction))
    for number, query in enumerate(NO_ANSWER):
        queries.append(dict(query=query, relevant=[], group=f"negative-{number}",
                            split="heldout" if number % 4 == 3 else "development",
                            language="ar" if number % 2 else "en"))
    return queries
