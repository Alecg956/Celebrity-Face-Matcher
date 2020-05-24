import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity import run_similarity_detector, plot_celeb_match


def test_basic_similarity():
    
    # load face input image
    inputSource = "/data/matching_input/test_similarity.jpg" 
    celeb_file = run_similarity_detector(inputSource)
    plot_celeb_match(inputSource, celeb_file)
    

def main():
    
    test_basic_similarity()
    

if __name__ == "__main__":
    main()


