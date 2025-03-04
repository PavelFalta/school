Feature: Bank Account

    Scenario: Vklad peněz na účet
        Given nový bankovní účet
        When vložím 200 Kč
        Then zůstatek je 200 Kč