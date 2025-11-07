
:- dynamic known/2.

suggest(bilgisayar_muhendisligi) :-
    symptom(likes_programming),
    symptom(likes_math).

suggest(psikoloji) :-
    symptom(likes_helping_people),
    symptom(likes_social).

suggest(tip) :-
    symptom(likes_helping_people),
    symptom(likes_bio),
    symptom(accepts_long_study).

suggest(mimarlik) :-
    symptom(likes_art),
    symptom(likes_buildings).

suggest(isletme) :-
    symptom(likes_business),
    \+ symptom(likes_math).

% Başlatıcı
start :-
    retractall(known(_, _)),
    findall(D, suggest(D), List),
    report(List).

report([]) :-
    writeln('Uygun bölüm bulunamadı.').
report(List) :-
    writeln('Senin için uygun olabilecek bölümler:'),
    forall(member(X, List), print_nice(X)).

print_nice(bilgisayar_muhendisligi) :- writeln('- Bilgisayar Mühendisliği').
print_nice(psikoloji) :- writeln('- Psikoloji').
print_nice(tip) :- writeln('- Tıp').
print_nice(mimarlik) :- writeln('- Mimarlık').
print_nice(isletme) :- writeln('- İşletme').

% Soru Sorma Mekanizması 
symptom(X) :-
    known(X, yes), !.
symptom(X) :-
    known(X, no), !, fail.
symptom(X) :-
    question(X, Y),
    format('~w (evet/hayır): ', [Y]),
    read_line_to_string(user_input, Input),
    normalize_answer(Input, Reply),
    (Reply = yes -> assertz(known(X, yes)) ; assertz(known(X, no)), fail).

% Sorular
question(likes_programming, 'Programlama ve bilgisayarla uğraşmaktan hoşlanır mısın?').
question(likes_math, 'Matematikle aranın iyi olduğunu düşünüyor musun?').
question(likes_helping_people, 'İnsanlara yardım etmekten hoşlanır mısın?').
question(likes_social, 'Sosyalleşmeyi sever misin?').
question(likes_bio, 'Biyolojiye ilgin var mı?').
question(accepts_long_study, 'Uzun eğitim sürecine razı mısın?').
question(likes_art, 'Sanat ve tasarımı sever misin?').
question(likes_buildings, 'Binalara, mimariye ilgin var mı?').
question(likes_business, 'İş dünyası, yönetim veya girişimcilik ilgini çeker mi?').

normalize_answer(Input, yes) :-
    memberchk(Low, ["evet","e","yes","y"]),
    string_lower(Input, Lower),
    sub_string(Lower, 0, _, _, Low), !.
normalize_answer(Input, no) :-
    memberchk(Low, ["hayır","hayir","h","no","n"]),
    string_lower(Input, Lower),
    sub_string(Lower, 0, _, _, Low), !.
normalize_answer(_, no) :-
    writeln('Anlaşılamadı, "hayır" olarak kabul ediyorum.').
