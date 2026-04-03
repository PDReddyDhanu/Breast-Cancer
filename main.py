import os
import sys


def run(command, msg):
    """
    Run system command safely
    """
    print("\n" + "="*50)
    print(msg)
    print("="*50)

    result = os.system(command)

    if result != 0:
        print("❌ Error while running:", command)
        sys.exit(1)

    print("✅ Done:", msg)


def main():

    print("\n🩺 AI Chemotherapy Prediction System")
    print("Starting Project Pipeline...\n")


    # Step 1: Generate Dataset
    run(
        "python dataset/generate_dataset.py",
        "STEP 1: Generating Dataset"
    )


    # Step 2: Train CNN (Side Effect)
    run(
        "python src/train_cnn.py",
        "STEP 2: Training CNN Model"
    )


    # Step 3: Train Regression (Severity)
    run(
        "python src/train_regression.py",
        "STEP 3: Training Severity Model"
    )


    # Step 4: Train Risk Classifier
    run(
        "python src/train_risk.py",
        "STEP 4: Training Risk Model"
    )


    # Step 5: Evaluate Models
    run(
        "python src/evaluate.py",
        "STEP 5: Evaluating Model Performance"
    )


    # Step 6: Launch UI
    print("\n🚀 Launching Web Interface...")
    print("Open browser at: http://127.0.0.1:7860\n")

    os.system("python interface/ui.py")


if __name__ == "__main__":
    main()
