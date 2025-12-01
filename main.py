from functions import inputFile, runAnalysis, runModel, exportResults
import numpy as np

if __name__ == '__main__':
    running = True
    file = None
    fileName = None
    filePath = None
    
    # Analysis results storage
    X_processed = None
    y_true = None
    df_features = None
    y_predictions = None

    print("-----------------------------------------")
    print("YapGPT - TCP Anomaly Detection System")
    print("-----------------------------------------")
    if fileName is None:
        print("[1] Upload Network Log File")
    else:
        print(f"[1] Upload Network Log File - {fileName}")
    print("[2] Run TCP Handshake Analysis")
    print("[3] View Detected Anomalies")
    print("[4] Export Results")
    print("[5] Exit")
    print("-----------------------------------------")  

    while(running):
        choice = input("Enter your choice (M to see menu): ").lower().strip()

        if choice == "m":
            print("-----------------------------------------")
            print("YapGPT - TCP Anomaly Detection System")
            print("-----------------------------------------")
            if fileName is None:
                print("[1] Upload Network Log File")
            else:
                print(f"[1] Upload Network Log File - {fileName}")
            print("[2] Run TCP Handshake Analysis")
            print("[3] View Detected Anomalies")
            print("[4] Export Results")
            print("[5] Exit")
            print("-----------------------------------------")
            continue

        
        if choice == '1':
            file, fileName = inputFile(file, fileName)
            if fileName:
                # Store the full file path that was entered by user
                # The inputFile function validates and opens the file, so we know path is valid
                # We need to reconstruct the path from what user entered
                filePath = fileName
                # Reset analysis state when new file is loaded
                X_processed = None
                y_true = None
                df_features = None
                y_predictions = None
                
        elif choice == '2':
            if fileName is None:
                print("\n✗ Please upload a network log file first (option 1)")
            else:
                print("\nStarting TCP Handshake Analysis...")
                # Run preprocessing and feature extraction
                X_processed, y_true, df_features = runAnalysis(filePath)
                
                if X_processed is not None:
                    # Run model prediction
                    y_predictions, y_true, df_features = runModel(X_processed, y_true, df_features)
                    
        elif choice == '3':
            if y_predictions is None:
                print("\n✗ No analysis results available. Please run analysis first (option 2)")
            else:
                print("\n" + "="*60)
                print("ANALYSIS RESULTS SUMMARY")
                print("="*60)
                print(f"  File analyzed: {filePath.split('/')[-1]}")
                print(f"  Total samples: {len(y_predictions):,}")
                
                # Show prediction distribution
                normal_count = np.sum(y_predictions == 0)
                attack_count = np.sum(y_predictions == 1)
                print(f"\n  Predicted Normal: {normal_count:,} ({normal_count/len(y_predictions)*100:.1f}%)")
                print(f"  Predicted Attack: {attack_count:,} ({attack_count/len(y_predictions)*100:.1f}%)")
                
                # Show accuracy if labels available
                if y_true is not None:
                    accuracy = np.mean(y_predictions == y_true)
                    print(f"\n  {'='*56}")
                    print(f"  MODEL ACCURACY: {accuracy*100:.2f}%")
                    print(f"  {'='*56}")
                    
                    # Per-class accuracy
                    print(f"\n  Detection Performance:")
                    for class_idx, class_name in enumerate(['Normal', 'Attack']):
                        mask = y_true == class_idx
                        if mask.sum() > 0:
                            correct = np.sum(y_predictions[mask] == y_true[mask])
                            total = mask.sum()
                            class_acc = (correct / total) * 100
                            print(f"    {class_name:8s}: {correct:,}/{total:,} correct ({class_acc:.2f}%)")
                    
                    # Confusion matrix summary
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true, y_predictions)
                    print(f"\n  Confusion Matrix:")
                    print(f"                  Predicted")
                    print(f"                Normal  Attack")
                    print(f"    Actual Normal  {cm[0,0]:6,}  {cm[0,1]:6,}")
                    print(f"           Attack  {cm[1,0]:6,}  {cm[1,1]:6,}")
                else:
                    print(f"\n  Note: No labels in file - accuracy cannot be calculated")
                    print(f"        Use a '_labeled.csv' file to see accuracy metrics")
                
                print(f"\n  Tip: Use option 4 to export visualization graphs")
                
        elif choice == '4':
            if y_predictions is None:
                print("\n✗ No results to export. Please run analysis first (option 2)")
            else:
                exportResults(y_predictions, y_true, df_features, fileName)
                
        elif choice == '5':
            print("\nExiting YapGPT - TCP Anomaly Detection System.")
            print("Thank you for using our system!")
            running = False
        else:
            print("Invalid input.")

