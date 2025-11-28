import click
import os
import numpy as np
from active_learning_manager import ActiveLearningManager

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    """Display main menu"""
    clear_screen()
    click.echo(" Teach Your Model - CLI App")
    click.echo("\n")
    click.echo("1. Get Prediction")
    click.echo("2. Submit Labels") 
    click.echo("3. Check Drift")
    click.echo("q. Quit")

@click.command()
def cli():
    """Interactive Teach Your Model CLI"""
    
    while True:
        show_menu()
        choice = click.prompt("Choose option", type=click.Choice(['1', '2', '3', 'q']))
        
        if choice == 'q':
            click.echo("Goodbye!")
            break
        elif choice == '1':
            get_prediction()
        elif choice == '2':
            submit_label() 
        elif choice == '3':
            check_drift()

def get_prediction():
    """Get prediction for text"""
    clear_screen()
    click.echo("Get Prediction")
    click.echo("\n" )
    
    manager = ActiveLearningManager()
    accuracy = manager.get_current_accuracy()
    click.echo(f"Current model accuracy: {accuracy:.3f}")
    click.echo("")
    
    while True:
        click.echo("Enter your text then press Enter twice:")
        
        lines = []
        while True:
            try:
                line = input()
                if line == "":  
                    break
                lines.append(line)
            except EOFError:
                break
        
        text = '\n'.join(lines).strip()
        
        if not text:
            click.echo("No text entered.")
            break
            
        result = manager.get_prediction(text)
        click.echo(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
        
        if result['action'] == 'logged':
            click.echo("Low confidence - logged for labeling")
        elif result['action'] == 'ask_user':
            if click.confirm("Is this correct?"):
                click.echo("Yay!")
            else:
                click.echo("Incorrect - logged for labeling")   
                manager._add_unlabeled(text,None, np.array([None]))

        click.echo("")
        if not click.confirm("Continue with another text?"):
            return
    

def submit_label():
    """Submit labels for logged samples"""
    clear_screen()
    click.echo("Submit Labels")
    click.echo("\n")
    
    manager = ActiveLearningManager()
    unlabeled_data = manager.get_unlabeled_samples()
    
    if not unlabeled_data:
        click.echo("No samples need labeling")
        click.prompt("Press Enter to continue...")
        return
    
    click.echo(f"Found {len(unlabeled_data)} samples needing labels:")
    click.echo("")
    
    labeled_count = 0
    label_map = {'t': 0, 's': 1, 'r': 2, 'p': 3, 'f': 4}
    class_names = ['technology', 'science', 'recreation', 'politics', 'forsale']
    
    for sample in unlabeled_data:
        click.echo(f"Text: {sample['text'][:800]}...")
        
        choice = click.prompt(
            "Label? [t]ech, [s]cience, [r]ec, [p]olitics, [f]orsale, [q]uit",
            type=click.Choice(['t', 's', 'r', 'p', 'f', 'q'])
        )
        
        if choice == 'q':
            return
        
        human_label = label_map[choice]
        manager.update_label(sample['text'], human_label)
        labeled_count += 1
        click.echo(f"Labeled as: {class_names[human_label]}")
        click.echo("")
    
    if labeled_count > 0:
        result = manager.check_retrain()
        click.echo(f"Labeled {labeled_count} samples. {result}")
        choice = click.prompt(
            "[q]uit ?",
            type=click.Choice(['q'])
        )
        
        if choice == 'q':
          return
    
    else:
        click.echo("No labels submitted")
    

def check_drift():
    """Check for model drift"""
    clear_screen()
    click.echo("Check Drift")
    click.echo("\n")
    
    manager = ActiveLearningManager()
    drift_result = manager.check_drift()
    click.echo(f"Drift Analysis: {drift_result}")

    choice = click.prompt(
            "[q]uit ?",
            type=click.Choice(['q'])
        )
        
    if choice == 'q':
        return
    

if __name__ == '__main__':
    cli()