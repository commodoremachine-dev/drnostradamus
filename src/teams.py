"""Takım adlarını tek bir standart ada çevirir (kaynaklar arasında yazım farklı)."""
import unicodedata

# Anahtar: aksansız küçük harf ad, değer: standart ad
_RAW_ALIASES = {
    "Atiker Konyaspor": "Konyaspor",
    "Aytemiz Alanyaspor": "Alanyaspor",
    "Gençlerbirligi": "Gençlerbirliği",
    "Kasimpasa": "Kasımpaşa",
    "Kasımpaşa SK": "Kasımpaşa",
    "Gazişehir Gaziantep FK": "Gaziantep FK",
    "BB Erzurumspor": "Erzurumspor",
    "MKE Ankaragücü": "Ankaragücü",
    "İstanbul Başakşehir": "Başakşehir",
    # football-data.co.uk yazımları (o kaynaktan CSV eklenirse)
    "Buyuksehyr": "Başakşehir", "Basaksehir": "Başakşehir",
    "Besiktas": "Beşiktaş", "Fenerbahce": "Fenerbahçe",
    "Rizespor": "Çaykur Rizespor", "Ankaragucu": "Ankaragücü",
    "Goztepe": "Göztepe", "Karagumruk": "Fatih Karagümrük",
    "Genclerbirligi": "Gençlerbirliği", "Eyupspor": "Eyüpspor",
    "Gaziantep": "Gaziantep FK", "Erzurum BB": "Erzurumspor",
    "Erzurumspor": "Erzurumspor", "Bodrumspor": "Bodrum FK",
    "Istanbulspor": "İstanbulspor AŞ", "Ad. Demirspor": "Adana Demirspor",
    "Yeni Malatyaspor": "Yeni Malatyaspor", "Akhisar": "Akhisar Belediyespor",
    # API-Football yazımları
    "Istanbul Basaksehir": "Başakşehir", "Kasimpasa": "Kasımpaşa",
    "Amed SK": "Amed", "Amedspor": "Amed", "Corum FK": "Çorum", "Corum": "Çorum",
    "Erzurumspor FK": "Erzurumspor", "Bodrumspor": "Bodrum FK", "Bodrum": "Bodrum FK",
    "Istanbulspor": "İstanbulspor AŞ", "MKE Ankaragucu": "Ankaragücü",
    "Yeni Malatyaspor": "Yeni Malatyaspor", "BB Erzurumspor": "Erzurumspor",
    "Kayserispor": "Kayserispor", "Gazisehir Gaziantep": "Gaziantep FK",
    # livescore (ESPN) yazımları
    "Amed SFK": "Amed", "Çorum FK": "Çorum", "Caykur Rizespor": "Çaykur Rizespor",
    "Altay Izmir": "Altay", "Umraniyespor": "Ümraniyespor", "Osmanlispor": "Osmanlıspor",
}
# Bilinen standart adlar: aksansız yazımları da bunlara eşlenir
CANONICAL = {
    "Adana Demirspor", "Akhisar Belediyespor", "Alanyaspor", "Amed", "Ankaragücü",
    "Antalyaspor", "Başakşehir", "Beşiktaş", "Bodrum FK", "Bursaspor", "Denizlispor",
    "Erzurumspor", "Eyüpspor", "Fatih Karagümrük", "Fenerbahçe", "Galatasaray",
    "Gaziantep FK", "Gençlerbirliği", "Göztepe", "Hatayspor", "Kasımpaşa", "Kayserispor",
    "Kocaelispor", "Konyaspor", "Pendikspor", "Samsunspor", "Sivasspor", "Trabzonspor",
    "Yeni Malatyaspor", "Çaykur Rizespor", "Çorum", "İstanbulspor AŞ",
    "Adanaspor", "Altay", "Gaziantepspor", "Giresunspor", "Karabükspor", "Osmanlıspor", "Ümraniyespor",
}
ALIASES: dict[str, str] = {}


def _key(name: str) -> str:
    s = name.strip().replace("ı", "i").replace("İ", "I")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


ALIASES.update({_key(c): c for c in CANONICAL})
ALIASES.update({_key(k): v for k, v in _RAW_ALIASES.items()})


def normalize(name: str) -> str:
    name = " ".join(name.split())
    return ALIASES.get(_key(name), name)
