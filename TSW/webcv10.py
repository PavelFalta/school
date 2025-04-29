from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/validate')
def validate_passenger():
    try:
        age_str = request.args.get('age')
        citizenship = request.args.get('citizenship')
        document_type = request.args.get('document_type')

        errors = []
        age = None # Initialize age to prevent UnboundLocalError

        
        if age_str is None:
            errors.append("Missing 'age' parameter.")
        if citizenship is None:
            errors.append("Missing 'citizenship' parameter.")
        

        if errors:
             return jsonify({"valid": False, "errors": errors}), 400

        
        # Only attempt age conversion if age_str was provided
        if age_str is not None:
            try:
                age = int(age_str)
                if not (0 <= age <= 120):
                    errors.append("Invalid age. Must be between 0 and 120.")
            except ValueError:
                errors.append("Invalid age format. Must be an integer.")

        
        valid_citizenships = ['EU', 'non-EU']
        if citizenship not in valid_citizenships:
            errors.append(f"Invalid citizenship. Must be one of: {valid_citizenships}.")

        
        if age is not None and age >= 2:
            if document_type is None:
                 errors.append("Missing 'document_type' parameter for age >= 2.")
            else:
                valid_document_types = ['passport', 'ID card']
                if document_type not in valid_document_types:
                    errors.append(f"Invalid document type. Must be one of: {valid_document_types}.")
                elif citizenship == 'non-EU' and document_type != 'passport':
                    errors.append("Non-EU citizens must have a passport.")
                

        elif age is not None and age < 2 and document_type is not None:
             errors.append("Document type should not be provided for children under 2.")


        if errors:
            return jsonify({"valid": False, "errors": errors}), 400
        else:
            return jsonify({"valid": True, "message": "Passenger data is valid."}), 200

    except Exception as e:
        
        return jsonify({"valid": False, "errors": [f"An unexpected error occurred: {str(e)}"]}), 500

if __name__ == '__main__':
    app.run(debug=True)
