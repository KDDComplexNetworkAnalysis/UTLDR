from enum import Enum

__author__ = ["Giulio Rossetti", "Letizia Milli", "Salvatore Citraro"]
__license__ = "BSD-2-Clause"


class Education(Enum):
    School = "school"
    University = "university"


class Ateco(Enum):
    Agricoltura = "A"
    Miniere = "B"
    Manifattura = "C"
    Energia = "D"
    AcqueRifiuti = "E"
    Edilizia = "F"
    Ingrosso = "G"
    Trasporto = "H"
    AlloggioRistorazione = "I"
    InformazioneComunicazione = "J"
    FinanzaAssicurazioni = "K"
    Immobiliari = "L"
    ScienzaTecnologia = "M"
    ServiziImprese = "N"
    PA_Difesa = "O"
    Istruzione = "P"
    Sanita = "Q"
    IntrattenimentoSport = "R"
    Altro = "S"


class Sociality(Enum):
    Normal = 0
    Quarantine = 1
    Lockdown = 2


class Weekdays(Enum):
    Monday = 1
    Tuesday	= 2
    Wednesday = 3
    Thursday = 4
    Friday = 5
    Saturday = 6
    Sunday = 7


class Regions(Enum):
    Piemonte = 1
    ValleAosta = 2
    Lombaridia = 3
    TrentinoAltoAdige = 4
    Veneto = 5
    FriuliVeneziaGiulia = 6
    Liguria = 7
    EmiliaRomagna = 8
    Toscana = 9
    Umbria = 10
    Marche = 11
    Lazio = 12
    Abruzzo = 13
    Molise = 14
    Campania = 15
    Puglia = 16
    Basilicata = 17
    Calabria = 18
    Sicilia = 19
    Sardegna = 20





