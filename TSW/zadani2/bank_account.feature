Feature: Bank Account

    Scenario: Vklad peněz na účet
        Given nový bankovní účet
        When vložím 200 Kč
        Then zůstatek je 200 Kč
    
    Scenario: Výběr peněz
        Given bankovní účet s 200 Kč
        When vyberu 100 Kč
        then zůstatek je 100 Kč