def remove_duplicate_links(input_file, output_file, summary_file):
    try:
        with open(input_file, 'r') as file:
            links = file.readlines()

        seen = set()
        unique_links = []
        duplicate_count = 0

        for link in links:
            stripped_link = link.strip()
            if stripped_link[-4:] not in seen:
                unique_links.append(stripped_link)
                seen.add(stripped_link[-4:])
            else:
                duplicate_count += 1

        with open(output_file, 'w') as file:
            for link in unique_links:
                file.write(link + '\n')
        
        with open(summary_file, 'w') as file:
                file.write(f"Total duplicates found: {duplicate_count}")

        print(f"Duplicates removed. Cleaned list saved to {output_file}.")
        print(f"Duplicate count: {duplicate_count}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return -1


input_file = "../data/raw/lead_links.csv"  # input file name
output_file = "../data/cleaned/cleaned_lead_links.csv"  # Output file name
output_summary_file = "../data/cleaned/summary.csv"  # Output file name

remove_duplicate_links(input_file, output_file, output_summary_file)